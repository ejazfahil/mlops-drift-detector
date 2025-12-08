"""PSI v2 2025-12-08"""
import numpy as np
def compute_psi(exp,act,n_bins=10,eps=1e-6):
    if np.std(exp)<eps or np.std(act)<eps: return 0.0
    b=np.linspace(np.min(exp),np.max(exp),n_bins+1);b[0]-=eps;b[-1]+=eps
    e=np.clip(np.histogram(exp,bins=b)[0]/len(exp),eps,None)
    a=np.clip(np.histogram(act,bins=b)[0]/len(act),eps,None)
    return float(np.sum((a-e)*np.log(a/e)))
def interpret_psi(v): return "stable" if v<0.10 else("moderate_shift" if v<0.25 else"significant_shift")
