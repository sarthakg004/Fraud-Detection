import numpy as np
import pandas as pd

from src.data import ID, TARGET

SNAPSHOT = pd.Timestamp("2023-08-05")  # data extraction date, inferred in notebook 01 section 8
# Day counters that stop when the bank closes / freezes an account (notebook 01 section 8). Every mule is frozen, so they
# encode the bank's *reaction*, not mule behaviour; a live monitor scores open accounts where they carry no signal -> leakage.
COUNTERS = ["others_6", "others_7", "others_9", "others_10", "others_12", "others_13", "others_15", "others_16", "demog_23"]
INCOME = ["0", "0 to 1L", "100001 to 5L", "5L to 10L", "10L to 25L", "25L to 50L", "50L to 3Crore", "3Crore to 5Crore",
          "5Crore to 10Crore", "10Crore to 15Crore", "15Crore to 20Crore", "20Crore to 35Crore"]
ORDINAL = {"income": {v: i for i, v in enumerate(INCOME)},
           "city_tier": {**{f"Tier {i}": i for i in range(1, 9)}, "Rural": 9},
           "demog_40": {"low": 0, "medium": 1, "High": 2}, "demog_43": {"low": 0, "medium": 1, "High": 2},
           "demog_22": {"N": 0, "Y": 1}}
CATEGORICAL = ["demog_2", "demog_4", "country_code", "occupation", "os", "email_domain"]


class FeatureBuilder:
    """Leak-free cleaning + engineered features selected in notebook 03.

    `fit` learns what must be identical at scoring time: constant columns and category levels.
    """

    def fit(self, df: pd.DataFrame):
        self.constant_ = [c for c in df.columns if df[c].nunique() <= 1]
        self.categories_ = {c: pd.Index(sorted(df[c].dropna().unique())) for c in CATEGORICAL}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.drop(columns=[ID, TARGET, *self.constant_, *COUNTERS], errors="ignore")
        X[["others_27", "others_28"]] = X[["others_27", "others_28"]].replace(9999, np.nan)  # "not applicable" code
        for c, mapping in ORDINAL.items():
            X[c] = X[c].map(mapping).astype(float)
        for c in CATEGORICAL:
            X[c] = pd.Categorical(X[c], categories=self.categories_[c])  # unseen levels -> NaN
        X["account_age"] = (SNAPSHOT - X.account_opening_date).dt.days
        X["open_dow"] = X.account_opening_date.dt.dayofweek
        return X.drop(columns="account_opening_date")

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
