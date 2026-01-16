import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
import gc
from torch.amp import autocast, GradScaler 
from transformers import BitsAndBytesConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from transformers import AutoModel, AutoConfig
from data_loader_nf import Dataset_ETT_hour 
from torch.utils.data import DataLoader
from metrics import metric, MSE

# ==========================================
# Config Template
# ==========================================
class Config:
    seq_len = 336 
    pred_len = 96 
    enc_in = 7
    c_out = 7
    patch_len = 16
    stride = 8 
    root_path = './data' 
    data_path = 'ETTm2.csv'
    model_save_path = './best_model.pt' 
    context_dim = 128 
    latent_dim = 64
    num_flows = 8 
    prefix_len = 16
    llm_model_name = 'meta-llama/Meta-Llama-3.1-8B'

    batch_size = 16
    accumulation_steps = 8
    
    learning_rate = 1e-4 
    epochs = 15     
    num_workers = 4   
    
    NF_lambda = 0.0 
    dropout = 0.1 

args = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Sub-Modules
# ==========================================
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine: self._init_params()
    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine: x = x * self.affine_weight + self.affine_bias
        return x
    def _denormalize(self, x):
        if self.affine: x = (x - self.affine_bias) / (self.affine_weight + 1e-10)
        x = x * self.stdev + self.mean
        return x
    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x); x = self._normalize(x)
        elif mode == 'denorm': x = self._denormalize(x)
        return x

class PlanarFlow(nn.Module):
    def __init__(self, z_dim):
        super(PlanarFlow, self).__init__()
        self.u = nn.Parameter(torch.randn(1, z_dim))
        self.w = nn.Parameter(torch.randn(1, z_dim))
        self.b = nn.Parameter(torch.randn(1))
    def forward(self, z):
        uw = torch.mm(self.u, self.w.t())
        m_factor = (-1 + F.softplus(uw))
        w_norm = torch.sum(self.w**2)
        u_hat = self.u + (m_factor * self.w / (w_norm + 1e-6))
        w_T_z = torch.mm(z, self.w.t()) 
        h = torch.tanh(w_T_z + self.b)
        f_z = z + torch.mm(h, u_hat)
        psi = (1 - h**2) * self.w 
        log_det_jacob = torch.log(torch.abs(1 + torch.mm(psi, u_hat.t())) + 1e-6)
        return f_z, log_det_jacob.squeeze() 

class NF_Module(nn.Module):
    def __init__(self, z_dim, h_dim, x_dim, num_flows):
        super(NF_Module, self).__init__()
        self.z_dim = z_dim
        self.mu_h = nn.Linear(h_dim, z_dim)
        self.logvar_h = nn.Linear(h_dim, z_dim)
        self.flows = nn.ModuleList([PlanarFlow(z_dim) for _ in range(num_flows)])
        self.reconstruct = nn.Sequential(
            nn.Linear(z_dim + h_dim, 256), nn.GELU(), nn.Dropout(args.dropout),
            nn.Linear(256, 512), nn.GELU(), nn.Dropout(args.dropout),
            nn.Linear(512, x_dim)
        )
    def forward(self, h, N_SAMPLES=1):
        mu = self.mu_h(h)
        logvar = self.logvar_h(h)
        std = torch.exp(0.5 * logvar)
        recon_samples = []
        log_det_Js = []
        for _ in range(N_SAMPLES):
            eps = torch.randn_like(std)
            z_0 = mu + eps * std 
            log_det_J = 0
            z_k = z_0
            for flow in self.flows:
                z_k, log_det = flow(z_k)
                log_det_J += log_det 
            log_det_Js.append(log_det_J) 
            inp = torch.cat([z_k, h], dim=1) 
            recon = self.reconstruct(inp)
            recon_samples.append(recon)
        return mu, logvar, torch.stack(recon_samples, dim=0), torch.stack(log_det_Js, dim=0).mean(dim=0)

# ==========================================
# Proposed Model 
# ==========================================
class ProposedModel(nn.Module):
    def __init__(self, args):
        super(ProposedModel, self).__init__()
        self.args = args
        self.revin = RevIN(1) 
        
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.num_patches = int((args.seq_len - self.patch_len) / self.stride) + 1
        
        self.input_dim = self.patch_len * 1 
        self.ts_encoder = nn.Linear(self.input_dim, args.context_dim)
        self.shortcut = nn.Linear(args.seq_len, args.pred_len)

        print(f"[INFO] Loading LLM: {args.llm_model_name} (4-bit)...")
        try:
            # 4-bit
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            config = AutoConfig.from_pretrained(args.llm_model_name)
            config.use_cache = False 
            
            self.llm = AutoModel.from_pretrained(
                args.llm_model_name, 
                config=config, 
                quantization_config=nf4_config,
                device_map='auto'
            )
            self.d_llm = config.hidden_size 
            self.llm.gradient_checkpointing_enable()
            
        except Exception as e:
            print(f"Error loading LLM: {e}")
            self.llm = None; self.d_llm = 768
            
        if self.llm:
            for param in self.llm.parameters(): param.requires_grad = False
        
        self.prefix = nn.Parameter(torch.randn(1, args.prefix_len, self.d_llm))
        
        self.reprogram = nn.Sequential(
            nn.Linear(args.context_dim, self.d_llm // 2),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.d_llm // 2, self.d_llm)
        )
        self.llm_out_proj = nn.Linear(self.d_llm, args.context_dim) 
        self.fusion_layer = nn.Sequential(
            nn.Linear(args.context_dim * 2, args.context_dim),
            nn.Dropout(args.dropout)
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.flat_target_dim = args.pred_len * 1
        self.nf_module = NF_Module(args.latent_dim, args.context_dim, self.flat_target_dim, args.num_flows)

    def forward(self, x_enc, x_target=None, text_list=None, N_SAMPLES=1):
        B, L, C = x_enc.shape # Batch, Length, Channel
        
        # 1. Input Normalization & Flattening
        x_enc = x_enc.permute(0, 2, 1).reshape(B * C, L, 1) 
        x_enc = self.revin(x_enc, 'norm') 
        shortcut_out = self.shortcut(x_enc.permute(0, 2, 1)).permute(0, 2, 1)

        # 2. Patching
        patches = x_enc.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # patches: (B*C, Num_Patches, Patch_Len)
        
        patches = patches.reshape(B*C, self.num_patches, self.patch_len)
        
        Z_t = self.ts_encoder(patches) # (B*C, Num_Patches, Context_Dim)
        
        # 3. LLM Processing
        if self.llm:
            # (B*C, Num_Patches, D_llm)
            llm_input_embeds = self.reprogram(Z_t) 
            

            prefix_expanded = self.prefix.expand(llm_input_embeds.shape[0], -1, -1)
            
 
            full_llm_input = torch.cat([prefix_expanded, llm_input_embeds], dim=1)
            
            # LLM Forward
            llm_outputs = self.llm(inputs_embeds=full_llm_input.to(dtype=torch.float16))
            hidden_states = llm_outputs.last_hidden_state.to(dtype=torch.float32)
            
            # (B*C, Num_Patches, D_llm)
            valid_out = hidden_states[:, args.prefix_len:, :]
            
            E_t = self.llm_out_proj(valid_out) # (B*C, Num_Patches, Context_Dim)
            
        else:
            E_t = torch.zeros_like(Z_t)
            
        # 4. Fusion
        concat_h = torch.cat([Z_t, E_t], dim=-1)
        context_h = self.fusion_layer(concat_h) 
        
        # Pooling: (B*C, Num_Patches, Dim) -> (B*C, Dim)
        context_h = context_h.permute(0, 2, 1) # (B*C, Dim, Num_Patches)
        context_h = self.pooling(context_h).squeeze(-1) 
        
        # 5. NF Decoding
        mu, logvar, recon_samples, log_det_J = self.nf_module(context_h, N_SAMPLES)
        recon_samples = recon_samples.reshape(B * C, -1, self.args.pred_len, 1)
        
        final_out = shortcut_out.unsqueeze(1) + recon_samples
        
        if self.training:
            recon_final = self.revin(final_out[:, 0, :, :], 'denorm')
            recon_final = recon_final.reshape(B, C, -1).permute(0, 2, 1)
            return recon_final, mu, logvar, log_det_J
        else:
            denorm_samples = []
            for i in range(final_out.shape[1]):
                 denorm_samples.append(self.revin(final_out[:, i, :, :], 'denorm'))
            stacked = torch.stack(denorm_samples, dim=1)
            stacked = stacked.reshape(B, C, -1, self.args.pred_len).permute(0, 2, 3, 1)
            return stacked

def move_learnable_to_device(model, device):
    for name, module in model.named_children():
        if name == 'llm': continue
        module.to(device)
    if hasattr(model, 'prefix'):
        model.prefix.data = model.prefix.data.to(device)

# ==========================================
# Main Experiment Loop
# ==========================================
if __name__ == '__main__':
    horizons = [96, 192, 336, 720]
    
    for pred_len in horizons:
        print(f"\n{'='*50}")
        print(f" >>> [START] Training for Prediction Length: {pred_len}")
        print(f"{'='*50}")
        
        args.pred_len = pred_len
        args.model_save_path = f'./best_model_len{pred_len}.pt'
        
        train_dataset = Dataset_ETT_hour(root_path=args.root_path, data_path=args.data_path, flag='train', size=[args.seq_len, 0, args.pred_len], features='M')
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            drop_last=True,
            pin_memory=True 
        )
        test_dataset = Dataset_ETT_hour(root_path=args.root_path, data_path=args.data_path, flag='test', size=[args.seq_len, 0, args.pred_len], features='M')
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers, 
            drop_last=False,
            pin_memory=True
        )

        model = ProposedModel(args)
        move_learnable_to_device(model, device)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scaler = GradScaler('cuda')
        best_metric = float('inf') 
        
        for epoch in range(args.epochs):
            t0 = time.time()
            model.train()
            optimizer.zero_grad() 
            
            for i, (batch_x, batch_y, batch_text) in enumerate(train_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                with autocast('cuda'):
                    recon, mu, logvar, log_det_J = model(batch_x, batch_y, text_list=None)
                    y_true = batch_y[:, -args.pred_len:, :]
                    loss = F.mse_loss(recon, y_true)
                    loss = loss / args.accumulation_steps 
                
                scaler.scale(loss).backward()
                
                if (i + 1) % args.accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                if (i+1) % 500 == 0:
                    print(f"\t[Epoch {epoch+1}] Iter {i+1} | Loss: {loss.item() * args.accumulation_steps:.4f}")
            
            scheduler.step()
            train_time = time.time() - t0
            
            should_validate_full = ((epoch + 1) % 5 == 0) or ((epoch + 1) == args.epochs)
            
            model.eval()
            preds_all = []
            trues_all = []
            preds_samples_all = [] 
            
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_text) in enumerate(test_loader):
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    y_true = batch_y[:, -args.pred_len:, :].cpu().numpy()
                    
                    n_samples = 10 if should_validate_full else 1
                    preds = model(batch_x, None, text_list=None, N_SAMPLES=n_samples)
                    preds_numpy = preds.cpu().numpy() 
                    
                    preds_mean = preds_numpy.mean(axis=1) 
                    preds_all.append(preds_mean)
                    trues_all.append(y_true)
                    preds_samples_all.append(preds_numpy)

            preds_all = np.concatenate(preds_all, axis=0)
            trues_all = np.concatenate(trues_all, axis=0)
            preds_samples_all = np.concatenate(preds_samples_all, axis=0) 
            preds_samples_all = preds_samples_all.transpose(1, 0, 2, 3)

            if should_validate_full:
                 mae, mse, rmse, mape, mspe, rse, corr, crps = metric(preds_all, trues_all, preds_samples_all)
                 print(f"Epoch {epoch+1} ({train_time:.1f}s) | [FULL] MSE: {mse:.4f} | MAE: {mae:.4f} | CRPS: {crps:.4f}")
            else:
                 mse = MSE(preds_all, trues_all)
                 print(f"Epoch {epoch+1} ({train_time:.1f}s) | [FAST] MSE: {mse:.4f}")

            if mse < best_metric:
                best_metric = mse
                torch.save(model.state_dict(), args.model_save_path)
                print(f" -> New Best Model Saved (MSE: {best_metric:.4f})")
        
        print(f"Done. Best MSE for Len {pred_len}: {best_metric:.4f}")
        
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        gc.collect()
