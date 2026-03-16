"""
Two-phase optimizer for pitcher projection model.

Phase 1: Bayesian optimization (Optuna) over hyperparams + individual features.
          Uses feature groups as toggles to keep the search space tractable,
          then saves the best config for phase 2.

Phase 2: Feature pruning. Takes the best config from Phase 1, ranks individual
          features by importance, iteratively drops the least useful ones,
          and checks whether removal improves CV score. Saves the final
          result as a loadable config for the Streamlit app.

Usage:
    python app/optimize.py                  # run both phases
    python app/optimize.py --trials 500     # more trials = better exploration
    python app/optimize.py --phase 1        # just hyperparams + groups
    python app/optimize.py --phase 2        # just feature pruning (uses saved best params)
"""

import argparse
import json
from pathlib import Path

import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "optimization"
CONFIGS_DIR = Path(__file__).resolve().parent.parent / "data" / "configs"

# ── Load the full feature list from the notebook pipeline ─────
with open(DATA_DIR / "feature_columns.json") as f:
    ALL_FEATURES: list[str] = json.load(f)

# Feature groups — used in Phase 1 to keep the search space manageable.
# Phase 2 then prunes at the individual feature level.
PITCHES = ["FA", "FT", "FC", "FS", "FO", "SI", "SL", "CU", "KC", "CH", "SC", "KN"]

FEATURE_GROUPS = {
    "Velocity": [f"v{p} (sc)" for p in PITCHES],
    "Horizontal Movement": [f"{p}-X (sc)" for p in PITCHES],
    "Vertical Movement": [f"{p}-Z (sc)" for p in PITCHES],
    "Batter Contact": ["ERA", "GB%", "Barrel%", "HR/9+"],
    "Batter Vision": ["K%", "BB%", "HR/FB", "SwStr%", "Contact% (sc)"],
    "X-Factor": ["Age", "FIP", "Clutch"],
}

LAG_ELIGIBLE_GROUPS = {"Velocity", "Horizontal Movement", "Vertical Movement"}


def lag_columns_for(base_cols: list[str]) -> list[str]:
    out = []
    for col in base_cols:
        out += [f"{col}_delta_1yr", f"{col}_t1", f"{col}_t2"]
    return out


def build_feature_list(selected_groups: list[str]) -> list[str]:
    features = []
    for group in selected_groups:
        base = FEATURE_GROUPS[group]
        features += base
        if group in LAG_ELIGIBLE_GROUPS:
            features += lag_columns_for(base)
    return features


def load_training_data(min_ip: int = 15):
    ps = pd.read_parquet(DATA_DIR / "processed_pitch_stats.parquet")
    train = ps.dropna(subset=["ERA_next"])
    train = train[train["IP"] > min_ip]
    max_season = train["Season"].max()
    train = train[train["Season"] < max_season]
    return train


def evaluate(train: pd.DataFrame, feature_cols: list[str], params: dict, cv: int = 5) -> float:
    """Return mean negative MAE across CV folds (higher = better)."""
    valid = [c for c in feature_cols if c in train.columns and train[c].notna().any()]
    if not valid:
        return -10.0

    X = train[valid]
    y = train["ERA_next"]

    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    return scores.mean()


# ── Phase 1: Bayesian optimization over hyperparams + feature groups ──

def phase1_objective(trial, train):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1500, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }

    selected = []
    for group in FEATURE_GROUPS:
        if trial.suggest_categorical(f"use_{group}", [True, False]):
            selected.append(group)
    if not selected:
        return -10.0

    min_ip = trial.suggest_int("min_ip", 10, 50, step=5)
    filtered = train[train["IP"] > min_ip]

    features = build_feature_list(selected)
    return evaluate(filtered, features, params)


def run_phase1(n_trials: int = 200):
    print(f"Phase 1: Bayesian optimization ({n_trials} trials)")
    print("=" * 60)

    train = load_training_data()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: phase1_objective(trial, train), n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\nBest CV MAE: {-best.value:.4f}")
    print(f"Best params: {json.dumps(best.params, indent=2)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "phase1_best.json", "w") as f:
        json.dump({"score": best.value, "params": best.params}, f, indent=2)

    df = study.trials_dataframe()
    df.to_csv(RESULTS_DIR / "phase1_trials.csv", index=False)

    return best.params


# ── Phase 2: Individual feature pruning ───────────────────────

def run_phase2():
    print("Phase 2: Feature importance pruning")
    print("=" * 60)

    best_path = RESULTS_DIR / "phase1_best.json"
    if not best_path.exists():
        print("Run phase 1 first (no phase1_best.json found)")
        return

    with open(best_path) as f:
        best = json.load(f)

    all_params = best["params"]

    xgb_params = {}
    selected_groups = []
    min_ip = 15
    for k, v in all_params.items():
        if k.startswith("use_"):
            if v:
                group_name = k[4:]
                selected_groups.append(group_name)
        elif k == "min_ip":
            min_ip = v
        else:
            xgb_params[k] = v

    train = load_training_data()
    train = train[train["IP"] > min_ip]
    all_features = build_feature_list(selected_groups)
    valid_features = [c for c in all_features if c in train.columns and train[c].notna().any()]

    baseline_score = evaluate(train, valid_features, xgb_params)
    print(f"Baseline MAE: {-baseline_score:.4f} ({len(valid_features)} features)")

    # Train once to get feature importances
    X = train[valid_features]
    y = train["ERA_next"]
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=valid_features).sort_values()

    current_features = list(valid_features)
    best_score = baseline_score
    best_features = list(valid_features)
    drop_batch_size = max(1, len(valid_features) // 20)

    print(f"Pruning in batches of {drop_batch_size}...")

    while len(current_features) > 5:
        to_drop = importances[importances.index.isin(current_features)].head(drop_batch_size).index.tolist()
        candidate = [f for f in current_features if f not in to_drop]

        score = evaluate(train, candidate, xgb_params)
        mae = -score
        print(f"  {len(candidate):3d} features -> MAE {mae:.4f}", end="")

        if score >= best_score:
            best_score = score
            best_features = candidate
            current_features = candidate
            print(" (improved)")
        else:
            print(" (worse, stopping)")
            break

    print(f"\nFinal: {len(best_features)} features, MAE {-best_score:.4f}")
    print(f"Dropped {len(valid_features) - len(best_features)} features")

    # Save optimization results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "xgb_params": xgb_params,
        "min_ip": min_ip,
        "features": best_features,
        "dropped_features": [f for f in valid_features if f not in best_features],
        "mae": -best_score,
        "n_features_original": len(valid_features),
        "n_features_final": len(best_features),
    }
    with open(RESULTS_DIR / "phase2_result.json", "w") as f:
        json.dump(result, f, indent=2)

    # Also save as a Streamlit-loadable config
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    save_config = {
        "features": best_features,
        "n_estimators": xgb_params.get("n_estimators", 500),
        "max_depth": xgb_params.get("max_depth", 6),
        "learning_rate": xgb_params.get("learning_rate", 0.07),
        "min_ip": min_ip,
        "season_to_predict": 2025,
    }
    with open(CONFIGS_DIR / "optimized.json", "w") as f:
        json.dump(save_config, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"Streamlit config saved as 'optimized' (load it from the app sidebar)")
    return result


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize pitcher projection model")
    parser.add_argument("--trials", type=int, default=200, help="Number of Optuna trials (phase 1)")
    parser.add_argument("--phase", type=int, choices=[1, 2], help="Run only this phase")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.phase is None or args.phase == 1:
        run_phase1(args.trials)
        print()

    if args.phase is None or args.phase == 2:
        run_phase2()
