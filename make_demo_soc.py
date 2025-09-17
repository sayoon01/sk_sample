import pandas as pd, numpy as np
from pathlib import Path
Path("raw_csv").mkdir(exist_ok=True)
t = pd.date_range("2024-06-01", periods=500, freq="5min")
rng = np.random.default_rng(0)
soc = (80 + np.cumsum(rng.normal(0,0.08,len(t)))).clip(0,100)
pd.DataFrame({"coll_dt": t, "soc": soc}).to_csv("raw_csv/demo_soc.csv", index=False)
print("saved raw_csv/demo_soc.csv")
