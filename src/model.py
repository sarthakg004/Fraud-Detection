import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data import TARGET
from src.features import FeatureBuilder

# data lives on CPU, booster on GPU -> xgboost warns on every predict; harmless at this size
warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")

XGB_PARAMS = dict(
    n_estimators=5000, learning_rate=0.05, max_depth=6, tree_method="hist", device="cuda",
    enable_categorical=True, eval_metric=["auc", "logloss"], early_stopping_rounds=200,
)


def evaluate(y, p) -> dict:
    """Ranking (ROC/PR-AUC) and probability-accuracy (log-loss, Brier) metrics."""
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p)}


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))[:, None]


class PlattCalibrator:
    """Logistic regression on logit(score). Maps weighted-loss scores back to true probabilities, keeps the ranking.
    Chosen in notebook 02 over prior correction (over-corrects) and isotonic (ties hurt PR-AUC)."""

    def fit(self, scores, y):
        self.lr_ = LogisticRegression(C=1e6).fit(_logit(scores), y)
        return self

    def predict(self, scores):
        return self.lr_.predict_proba(_logit(scores))[:, 1]


class XGBCV:
    """Stratified K-fold XGBoost trained with a class-weighted log-loss.

    Mule weight = (#legit / #mule) ** weight_power, i.e. tied to the class distribution:
    0 = unweighted, 0.5 = sqrt-balanced (x7 here), 1 = fully balanced (x49 here).
    The same weight is used in the early-stopping eval set, so learning curves show the objective actually optimised.
    Scores are on the weighted scale (inflated); calibrate before reading them as probabilities.
    `sampler(X, y) -> (X, y)` optionally resamples the *training* part of each fold (e.g. SMOTE);
    validation folds always keep the true class balance.
    """

    def __init__(self, params=None, n_splits=5, weight_power=1.0, seed=42, sampler=None):
        self.params = {**XGB_PARAMS, "random_state": seed, **(params or {})}
        self.n_splits, self.weight_power, self.seed, self.sampler = n_splits, weight_power, seed, sampler

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        self.pos_weight_ = ((y == 0).sum() / (y == 1).sum()) ** self.weight_power
        w = np.where(y == 1, self.pos_weight_, 1.0)
        self.oof_, self.models_, self.curves_ = np.zeros(len(y)), [], []
        folds = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed).split(X, y)
        for tr, va in folds:
            X_tr, y_tr = self.sampler(X.iloc[tr], y[tr]) if self.sampler else (X.iloc[tr], y[tr])
            w_tr = np.where(y_tr == 1, self.pos_weight_, 1.0)
            m = xgb.XGBClassifier(**self.params)
            m.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_tr, y_tr), (X.iloc[va], y[va])], sample_weight_eval_set=[w_tr, w[va]], verbose=False)
            self.oof_[va] = m.predict_proba(X.iloc[va])[:, 1]
            self.models_.append(m)
            self.curves_.append(m.evals_result())
        return self

    @property
    def best_iters_(self):
        return [m.best_iteration for m in self.models_]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Mean of the fold models' scores (weighted scale)."""
        return np.mean([m.predict_proba(X)[:, 1] for m in self.models_], axis=0)

    def feature_importance(self, kind="gain") -> pd.Series:
        imp = [pd.Series(m.get_booster().get_score(importance_type=kind)) for m in self.models_]
        return pd.concat(imp, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)


class FraudModel:
    """End-to-end mule scorer: FeatureBuilder -> K-fold weighted XGBoost (fold models averaged) -> Platt calibration.

    `fit` takes the raw development frame (with Target); `predict_proba` takes any raw frame and returns P(mule).
    """

    def __init__(self, params=None, weight_power=1.0, seed=42):
        self.params, self.weight_power, self.seed = params, weight_power, seed

    def fit(self, df: pd.DataFrame):
        y = df[TARGET].to_numpy()
        self.features_ = FeatureBuilder()
        X = self.features_.fit_transform(df)
        self.cv_ = XGBCV(self.params, weight_power=self.weight_power, seed=self.seed).fit(X, y)
        self.calibrator_ = PlattCalibrator().fit(self.cv_.oof_, y)
        self.oof_ = self.calibrator_.predict(self.cv_.oof_)  # honest out-of-fold P(mule) for every training row
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.calibrator_.predict(self.cv_.predict_proba(self.features_.transform(df)))

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "FraudModel":
        return joblib.load(path)
