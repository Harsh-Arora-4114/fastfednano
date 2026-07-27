"""
FAST-FedNano | Export trained models -> JSON (for Cloudflare Worker inference)
------------------------------------------------------------------------------
Run this LOCALLY, in the same folder as your Phase 1 outputs:
    preprocessor.joblib, model_random_forest.joblib, model_xgboost.joblib,
    model_neural_net.joblib, feature_names.json

It produces ONE file: model_bundle.json
Copy that file into ml-worker/models/model_bundle.json before deploying
the Cloudflare Worker (see README Step 2).

Why JSON and not the .joblib files directly:
    Cloudflare Workers run JavaScript (V8), not Python — they can't load
    scikit-learn/xgboost/joblib objects. So we extract the raw numbers
    (tree splits, leaf values, NN weights, scaler mean/std) into plain
    JSON, and re-implement prediction as pure-JS math in the Worker.

IMPORTANT ASSUMPTIONS (match these to how you actually trained):
  - preprocessor.joblib is a sklearn ColumnTransformer with a "num"
    branch (SimpleImputer + StandardScaler) as built in data_preprocessing.py.
    If you also have a "cat" (OneHotEncoder) branch because your real
    dataset has categorical columns, this script will warn you — the
    JS worker in this bundle only implements the numeric path.
  - model_xgboost.joblib was trained with objective="reg:squarederror"
    (the default for regression) — this has an IDENTITY link function,
    so prediction = base_score + sum(leaf values across trees), no
    sigmoid/softmax needed. If you used a different objective, tell me
    and I'll adjust the JS inference.

Usage:
    pip install scikit-learn xgboost joblib numpy --break-system-packages
    python export_model.py
"""

import json
import joblib
import numpy as np


def export_scaler(preprocessor):
    """Extract mean_/scale_ from the numeric StandardScaler branch."""
    named = preprocessor.named_transformers_
    if "cat" in named and hasattr(named["cat"], "named_steps"):
        onehot = named["cat"].named_steps.get("onehot")
        if onehot is not None and len(getattr(onehot, "categories_", [])) > 0:
            print(
                "[WARNING] Your preprocessor has a categorical (OneHotEncoder) "
                "branch. This export script + the JS worker only implement the "
                "numeric StandardScaler path. If your real dataset has "
                "categorical columns, ping me and I'll extend the JS inference "
                "to handle one-hot encoding too."
            )

    num_pipeline = named["num"]
    imputer = num_pipeline.named_steps["imputer"]
    scaler = num_pipeline.named_steps["scaler"]

    return {
        "impute_values": imputer.statistics_.tolist(),  # median per feature (fallback for missing values)
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }


def export_random_forest(model):
    """Export every tree in the forest as flat arrays (sklearn's own tree_ format)."""
    trees = []
    for est in model.estimators_:
        t = est.tree_
        trees.append({
            "children_left": t.children_left.tolist(),
            "children_right": t.children_right.tolist(),
            "feature": t.feature.tolist(),
            "threshold": t.threshold.tolist(),
            # regression leaf value lives at value[node][0][0]
            "value": [float(v[0][0]) for v in t.value],
        })
    return {"n_estimators": len(trees), "trees": trees}


def export_xgboost(model):
    """
    Use XGBoost's own JSON tree dump (per-node dicts with yes/no/children) —
    far simpler to walk in JS than the internal binary tree format.
    """
    booster = model.get_booster()
    dump = booster.get_dump(dump_format="json")
    trees = [json.loads(t) for t in dump]

    # base_score: additive baseline all trees are added on top of
    config = json.loads(booster.save_config())
    base_score = float(
        config["learner"]["learner_model_param"]["base_score"]
    )
    objective = config["learner"]["objective"]["name"]
    if objective != "reg:squarederror":
        print(
            f"[WARNING] Your XGBoost objective is '{objective}', not "
            "'reg:squarederror'. This export assumes an identity link "
            "(prediction = base_score + sum of leaves). If your objective "
            "needs a different link function (e.g. logistic), the JS "
            "inference will give wrong numbers — let me know and I'll fix it."
        )

    return {"base_score": base_score, "trees": trees}


def export_neural_net(model):
    """Export MLPRegressor weights — plain dense layers + ReLU (sklearn default)."""
    if model.activation != "relu":
        print(
            f"[WARNING] Your MLP activation is '{model.activation}', not "
            "'relu'. The JS worker in this bundle only implements ReLU — "
            "tell me if you used tanh/logistic and I'll add it."
        )
    return {
        "activation": model.activation,
        "weights": [w.tolist() for w in model.coefs_],
        "biases": [b.tolist() for b in model.intercepts_],
    }


def main():
    with open("feature_names.json") as f:
        meta = json.load(f)

    preprocessor = joblib.load("preprocessor.joblib")
    rf = joblib.load("model_random_forest.joblib")
    xgb = joblib.load("model_xgboost.joblib")
    nn = joblib.load("model_neural_net.joblib")

    bundle = {
        "target": meta["target"],
        "feature_names": meta["feature_names"],
        "scaler": export_scaler(preprocessor),
        "random_forest": export_random_forest(rf),
        "xgboost": export_xgboost(xgb),
        "neural_network": export_neural_net(nn),
    }

    with open("model_bundle.json", "w") as f:
        json.dump(bundle, f)

    import os
    size_mb = os.path.getsize("model_bundle.json") / (1024 * 1024)
    print(f"\nSaved model_bundle.json ({size_mb:.2f} MB)")
    if size_mb > 8:
        print(
            "[NOTE] Bundle is fairly large. Cloudflare Workers (free plan) "
            "allow up to 10MB per script+assets after compression, so this "
            "should still deploy fine, but if you add more trees/estimators "
            "later and it grows past that, tell me and I'll switch the "
            "Worker to load the bundle from Cloudflare R2/KV at request time "
            "instead of bundling it directly."
        )
    print("Copy this file to: ml-worker/models/model_bundle.json")


if __name__ == "__main__":
    main()
