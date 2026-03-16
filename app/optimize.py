"""
Two-phase optimizer for pitcher projection model.

Phase 1: Bayesian optimization (Optuna) over hyperparams + feature groups.
          Uses feature group toggles to keep the search space tractable (~200
          trials, ~7 min). Also explores which lag/engineering transforms help.

Phase 2: Feature pruning. Takes the best config from Phase 1, ranks individual
          features by importance, iteratively drops the least useful ones.
          Saves the result as a Streamlit-loadable config.

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

EXCLUDE_COLS = {"IDfg", "Season", "Name", "Team", "Age Rng"}

# ── Feature groups for Phase 1 search ─────────────────────────
PITCHES = ["FA", "FT", "FC", "FS", "FO", "SI", "SL", "CU", "KC", "CH", "SC", "KN"]

FEATURE_GROUPS = {
    "Velocity": [f"v{p} (sc)" for p in PITCHES],
    "Horizontal Movement": [f"{p}-X (sc)" for p in PITCHES],
    "Vertical Movement": [f"{p}-Z (sc)" for p in PITCHES],
    "Batter Contact": ["ERA", "GB%", "Barrel%", "HR/9+"],
    "Batter Vision": ["K%", "BB%", "HR/FB", "SwStr%", "Contact% (sc)"],
    "X-Factor": ["Age", "FIP", "Clutch"],
    "Stuff+": ["Stuff+", "Location+", "Pitching+"],
    "Batted Ball": ["Barrel%", "HardHit%", "EV", "LA", "Soft%", "Med%", "Hard%"],
    "Plate Discipline": ["O-Swing%", "Z-Swing%", "O-Contact%", "Z-Contact%", "Zone%", "SwStr%"],
}

ENGINEERING_OPTIONS = {
    "_t1": "shift(1)",
    "_t2": "shift(2)",
    "_delta_1yr": "delta",
    "_avg_3yr": "avg3",
}


def load_base_data():
    return pd.read_parquet(DATA_DIR / "pitching_stats.parquet")


def engineer_features(df, base_features, engineering):
    """Same logic as the Streamlit app."""
    ps = df.copy().sort_values(["IDfg", "Season"])
    all_feature_cols = list(base_features)

    for suffix, cols in engineering.items():
        for col in cols:
            if col not in ps.columns:
                continue
            if suffix == "_t1":
                ps[f"{col}{suffix}"] = ps.groupby("IDfg")[col].shift(1)
            elif suffix == "_t2":
                ps[f"{col}{suffix}"] = ps.groupby("IDfg")[col].shift(2)
            elif suffix == "_delta_1yr":
                t1 = ps.groupby("IDfg")[col].shift(1)
                ps[f"{col}{suffix}"] = ps[col] - t1
            elif suffix == "_avg_3yr":
                t1 = ps.groupby("IDfg")[col].shift(1)
                t2 = ps.groupby("IDfg")[col].shift(2)
                ps[f"{col}{suffix}"] = (ps[col] + t1 + t2) / 3
            all_feature_cols.append(f"{col}{suffix}")

    ps["ERA_next"] = ps.groupby("IDfg")["ERA"].shift(-1)
    return ps, all_feature_cols


def prepare_training_data(ps, feature_cols, min_ip):
    """Filter and prepare training data, returning X, y, valid_features."""
    train = ps.dropna(subset=["ERA_next"])
    train = train[train["IP"] > min_ip]
    max_season = train["Season"].max()
    train = train[train["Season"] < max_season]

    valid = [c for c in feature_cols if c in train.columns and train[c].notna().any()]
    return train, valid


def evaluate(train, valid_features, params, cv=5):
    """Return mean negative MAE across CV folds (higher = better)."""
    if not valid_features:
        return -10.0

    X = train[valid_features]
    y = train["ERA_next"]

    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    return scores.mean()


# ── Phase 1: Bayesian optimization ────────────────────────────

def phase1_objective(trial, base_df):
    # Hyperparameters
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

    # Feature group selection
    base_features = []
    for group_name, group_cols in FEATURE_GROUPS.items():
        if trial.suggest_categorical(f"use_{group_name}", [True, False]):
            base_features.extend(group_cols)
    if not base_features:
        return -10.0
    # Deduplicate (some features appear in multiple groups)
    base_features = list(dict.fromkeys(base_features))

    # Engineering toggles
    engineering = {}
    use_t1 = trial.suggest_categorical("eng_t1", [True, False])
    use_t2 = trial.suggest_categorical("eng_t2", [True, False])
    use_delta = trial.suggest_categorical("eng_delta", [True, False])
    use_avg3 = trial.suggest_categorical("eng_avg3", [True, False])

    # Apply engineering to velocity/movement features (most impactful for trends)
    eng_eligible = [f for f in base_features if "(sc)" in f and ("v" in f or "-X" in f or "-Z" in f)]
    if use_t1 and eng_eligible:
        engineering["_t1"] = eng_eligible
    if use_t2 and eng_eligible:
        engineering["_t2"] = eng_eligible
    if use_delta and eng_eligible:
        engineering["_delta_1yr"] = eng_eligible
    if use_avg3 and eng_eligible:
        engineering["_avg_3yr"] = eng_eligible

    min_ip = trial.suggest_int("min_ip", 10, 50, step=5)

    ps, all_feature_cols = engineer_features(base_df, base_features, engineering)
    train, valid_features = prepare_training_data(ps, all_feature_cols, min_ip)

    return evaluate(train, valid_features, params)


def run_phase1(n_trials=200):
    print(f"Phase 1: Bayesian optimization ({n_trials} trials)")
    print("=" * 60)

    base_df = load_base_data()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: phase1_objective(trial, base_df), n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\nBest CV MAE: {-best.value:.4f}")
    print(f"Best params: {json.dumps(best.params, indent=2, default=str)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "phase1_best.json", "w") as f:
        json.dump({"score": best.value, "params": best.params}, f, indent=2, default=str)

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

    # Separate XGBoost params from search params
    xgb_params = {}
    selected_groups = []
    min_ip = 15
    eng_flags = {}

    for k, v in all_params.items():
        if k.startswith("use_"):
            if v:
                selected_groups.append(k[4:])
        elif k == "min_ip":
            min_ip = v
        elif k.startswith("eng_"):
            eng_flags[k] = v
        else:
            xgb_params[k] = v

    # Rebuild base features from groups
    base_features = []
    for group_name in selected_groups:
        if group_name in FEATURE_GROUPS:
            base_features.extend(FEATURE_GROUPS[group_name])
    base_features = list(dict.fromkeys(base_features))

    # Rebuild engineering
    eng_eligible = [f for f in base_features if "(sc)" in f and ("v" in f or "-X" in f or "-Z" in f)]
    engineering = {}
    if eng_flags.get("eng_t1") and eng_eligible:
        engineering["_t1"] = eng_eligible
    if eng_flags.get("eng_t2") and eng_eligible:
        engineering["_t2"] = eng_eligible
    if eng_flags.get("eng_delta") and eng_eligible:
        engineering["_delta_1yr"] = eng_eligible
    if eng_flags.get("eng_avg3") and eng_eligible:
        engineering["_avg_3yr"] = eng_eligible

    base_df = load_base_data()
    ps, all_feature_cols = engineer_features(base_df, base_features, engineering)
    train, valid_features = prepare_training_data(ps, all_feature_cols, min_ip)

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

    # Separate final features into base vs engineered for config format
    final_base = [f for f in best_features if not any(f.endswith(s) for s in ["_t1", "_t2", "_delta_1yr", "_avg_3yr"])]
    final_eng = {}
    for f in best_features:
        for suffix in ["_t1", "_t2", "_delta_1yr", "_avg_3yr"]:
            if f.endswith(suffix):
                base_col = f[: -len(suffix)]
                final_eng.setdefault(suffix, [])
                if base_col not in final_eng[suffix]:
                    final_eng[suffix].append(base_col)

    # Save optimization results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "xgb_params": xgb_params,
        "min_ip": min_ip,
        "base_features": final_base,
        "engineering": final_eng,
        "all_features": best_features,
        "dropped_features": [f for f in valid_features if f not in best_features],
        "mae": -best_score,
        "n_features_original": len(valid_features),
        "n_features_final": len(best_features),
    }
    with open(RESULTS_DIR / "phase2_result.json", "w") as f:
        json.dump(result, f, indent=2)

    # Save as Streamlit-loadable config
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    streamlit_config = {
        "base_features": final_base,
        "engineering": final_eng,
        "n_estimators": xgb_params.get("n_estimators", 500),
        "max_depth": xgb_params.get("max_depth", 6),
        "learning_rate": xgb_params.get("learning_rate", 0.07),
        "min_ip": min_ip,
        "season_to_predict": 2025,
    }
    with open(CONFIGS_DIR / "optimized.json", "w") as f:
        json.dump(streamlit_config, f, indent=2)

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
