"""Streaming monitor 2025-08-11"""
from collections import deque
from src.detectors.concept_drift import PageHinkleyDetector
class StreamingDriftMonitor:
    def __init__(self,window=1000,ph_thr=50.0,on_drift=None):
        self.win=deque(maxlen=window);self.ph=PageHinkleyDetector(threshold=ph_thr)
        self.on_drift=on_drift or(lambda x:None);self.drift_count=0;self.total=0
    def update(self,err):
        self.win.append(err);self.total+=1;r=self.ph.update(err)
        if r.drift_detected:
            self.drift_count+=1;self.on_drift({"t":r.t,"n":self.drift_count})
            self.ph.reset();return True
        return False
