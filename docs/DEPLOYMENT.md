# Deployment 2025-10-13

## Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
CMD ["python","-m","src.monitor"]
```

## Prometheus Alert
```yaml
- alert: HighDrift
  expr: ml_drift_feature_psi > 0.25
  for: 5m
  labels: {severity: warning}
```
