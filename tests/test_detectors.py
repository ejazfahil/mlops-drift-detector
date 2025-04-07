"""Tests 2025-04-07"""
import sys,os;sys.path.insert(0,os.path.join(os.path.dirname(__file__),".."))  
import numpy as np,pandas as pd
from src.detectors.feature_drift import FeatureDriftDetector
np.random.seed(42)
def mk(mu=0): return pd.DataFrame({"f":np.random.normal(mu,1,500)})
def test_no_drift(): assert not FeatureDriftDetector(mk()).detect(mk())["overall"]
def test_drift(): assert FeatureDriftDetector(mk(mu=0),0.1).detect(mk(mu=10))["overall"]
