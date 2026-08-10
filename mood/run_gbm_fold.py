"""Train and evaluate two XGBoost regressors for one mood CV fold."""

import argparse
import json

import numpy as np
import xgboost as xgb

from mood.metrics import regression_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    spec = json.loads(open(args.spec).read())
    X_train, y_train = np.load(spec["X_train_path"]), np.load(spec["y_train_path"])
    X_val, y_val = np.load(spec["X_val_path"]), np.load(spec["y_val_path"])

    predictions = []
    for target_idx in range(2):
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
            random_state=spec.get("seed", 42), n_jobs=-1,
        )
        model.fit(X_train, y_train[:, target_idx])
        predictions.append(model.predict(X_val))
    metrics = regression_metrics(y_val, np.column_stack(predictions))
    with open(args.out, "w") as output:
        json.dump(metrics, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
