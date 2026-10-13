"""Train the final mule model on all development data and save it.  Usage: python -m src.train"""
from pathlib import Path

from src.data import TARGET, DataLoader
from src.model import FraudModel, evaluate

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

# Optuna winner, notebook 04 section 3: OOF log-loss 0.0129 vs 0.0156 with default params
XGB_TUNED = dict(learning_rate=0.03071, max_depth=9, min_child_weight=4.674, subsample=0.5015, colsample_bytree=0.5678,
                 reg_lambda=2.501, reg_alpha=0.003836, gamma=1.755, max_delta_step=0.6497)
# mule weight = (#legit / #mule) ** 0.2113 ~= 2.3; heavier weights lost in the notebook 04 sweep
WEIGHT_POWER = 0.2113


def main():
    train = DataLoader().load("train")
    model = FraudModel(XGB_TUNED, WEIGHT_POWER).fit(train)
    oof = evaluate(train[TARGET], model.oof_)
    print("5-fold out-of-fold metrics on all 100k accounts: " + ", ".join(f"{k} {v:.4f}" for k, v in oof.items()))
    print(f"best rounds per fold: {model.cv_.best_iters_} | mule weight {model.cv_.pos_weight_:.2f}")
    ARTIFACTS.mkdir(exist_ok=True)
    model.save(ARTIFACTS / "model.joblib")
    print(f"saved {ARTIFACTS / 'model.joblib'}")


if __name__ == "__main__":
    main()
