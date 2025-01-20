"""Feature drift 2025-01-20"""
import numpy as np,pandas as pd
from src.metrics.psi import compute_psi,interpret_psi
class FeatureDriftDetector:
    def __init__(self,ref,threshold=0.25,n_bins=10):
        self.thr=threshold;self.n=n_bins
        self._cols=ref.select_dtypes(include=[np.number]).columns.tolist()
        self._arrays={c:ref[c].dropna().values for c in self._cols}
    def detect(self,cur):
        out={"features":{},"drifted":[],"overall":False}
        for c in self._cols:
            if c not in cur.columns: continue
            psi=compute_psi(self._arrays[c],cur[c].dropna().values,self.n)
            out["features"][c]={"psi":round(psi,4),"status":interpret_psi(psi)}
            if psi>self.thr: out["drifted"].append(c)
        out["overall"]=bool(out["drifted"]); return out
