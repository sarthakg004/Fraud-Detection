from pathlib import Path

import pandas as pd

ID, TARGET = "Primary key", "Target"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class DataLoader:
    """Reads the raw Excel files once and caches them as parquet (xlsx parsing takes ~1 min)."""

    FILES = {"train": "train.xlsx", "validation": "validation.xlsx"}

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = Path(data_dir)

    def load(self, split: str) -> pd.DataFrame:
        cache = self.data_dir / f"{split}.parquet"
        if cache.exists():
            return pd.read_parquet(cache)
        df = pd.read_excel(self.data_dir / self.FILES[split], engine="openpyxl")
        # demog_2 / demog_4 / income mix ints and strings (e.g. 3, 'ZZ', 'N') -> one string type
        for c in df.columns[df.dtypes == object]:
            df[c] = df[c].where(df[c].isna(), df[c].astype(str))
        df.to_parquet(cache, index=False)
        return df
