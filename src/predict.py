"""Score the validation file with the saved model.  Usage: python -m src.predict"""
import numpy as np
import pandas as pd

from src.data import ID, DataLoader
from src.model import FraudModel
from src.train import ARTIFACTS


def main():
    model = FraudModel.load(ARTIFACTS / "model.joblib")
    valid = DataLoader().load("validation")
    p = model.predict_proba(valid)
    # sanity checks required by the case study
    assert len(p) == len(valid) == valid[ID].nunique() and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    out = ARTIFACTS / "submission.csv"
    pd.DataFrame({ID: valid[ID], "Probability": p}).to_csv(out, index=False)
    print(f"wrote {out} | {len(p):,} accounts | mean P(mule) {p.mean():.4f} | P>=0.5: {(p >= .5).sum():,}")


if __name__ == "__main__":
    main()
