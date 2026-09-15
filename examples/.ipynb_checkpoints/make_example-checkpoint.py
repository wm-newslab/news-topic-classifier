from pathlib import Path
import pandas as pd
path = Path("examples/input/demo/example.com/2026/01/01.parquet")
path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame([{"unique_id": "demo-1", "title": "City opens new bus route", "content": "A new bus route connects the downtown transit station with residential neighborhoods. Service starts Monday."}]).to_parquet(path, index=False)
print(path)
