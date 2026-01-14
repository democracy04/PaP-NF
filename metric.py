import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def CRPS(preds, true, normalize=True):
    """
    Lag-Llama(GluonTS Evaluator)와 동일한 계산 방식
    preds: (n_samples, batch, time, dim)
    true:  (batch, time, dim)
    """
    n_samples = preds.shape[0]
    
    # Term 1: E|X - y| (MAE part)
    # (n_samples, B, T, D) - (B, T, D)
    absolute_errors = np.abs(preds - true[None, ...])
    term1 = np.mean(absolute_errors, axis=0) # (B, T, D)

    if n_samples > 1:
        term2 = np.mean(np.abs(preds[:, None, ...] - preds[None, :, ...]), axis=(0, 1))
    else:
        term2 = 0

    crps_raw = term1 - 0.5 * term2

    total_crps = np.sum(crps_raw)
    total_abs_true = np.sum(np.abs(true)) + 1e-8
    
    weighted_crps = total_crps / total_abs_true
    return weighted_crps


def metric(pred, true, preds_samples=None):
    """
    pred: (Batch, Time, Dim) -> Deterministic mean prediction
    true: (Batch, Time, Dim) -> Ground Truth
    preds_samples: (Samples, Batch, Time, Dim) -> For CRPS
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    
    crps = 0.0
    if preds_samples is not None:
        crps = CRPS(preds_samples, true, normalize=True)
        
    return mae, mse, rmse, mape, mspe, rse, corr, crps
