"""Page-Hinkley 2025-02-24"""
from dataclasses import dataclass
@dataclass
class PHResult: drift_detected:bool; t:int; cumsum:float; threshold:float
class PageHinkleyDetector:
    def __init__(self,delta=0.005,threshold=50.0,alpha=1.0):
        self.delta=delta;self.threshold=threshold;self.alpha=alpha;self.reset()
    def reset(self): self._s=0.0;self._ms=float("inf");self._t=0;self._m=0.0
    def update(self,v):
        self._t+=1;self._m=self.alpha*self._m+(1-self.alpha)*v
        self._s+=v-self._m-self.delta;self._ms=min(self._ms,self._s)
        ph=self._s-self._ms
        return PHResult(ph>self.threshold,self._t,ph,self.threshold)
