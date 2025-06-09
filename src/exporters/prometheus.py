"""Prometheus 2025-06-09"""
class PrometheusExporter:
    def export(self,r):
        parts=[f'# TYPE ml_drift_feature_psi gauge\nml_drift_feature_psi{{feature="{f}"}} {s["psi"]}'
               for f,s in r.get("features",{}).items()]
        parts.append(f"ml_drift_overall_rate {r.get('rate',0)}")
        return "\n".join(parts)+"\n"
