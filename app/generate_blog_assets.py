"""Generate blog-ready charts for model evaluation across multiple years."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CONFIGS_DIR = Path(__file__).resolve().parent.parent / "data" / "configs"
OUT_DIR = Path(__file__).resolve().parent.parent / "blog_assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_COLS = {"IDfg", "Season", "Name", "Team", "Age Rng"}

ENGINEERING_SUFFIXES = ["_t1", "_t2", "_delta_1yr", "_avg_3yr"]


def load_base_data():
    return pd.read_parquet(DATA_DIR / "pitching_stats.parquet")


def engineer_features(df, base_features, engineering):
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


def train_and_predict(ps, feature_cols, season_to_predict, cfg):
    min_ip = cfg["min_ip"]
    train_data = ps.dropna(subset=["ERA_next"])
    train_data = train_data[train_data["Season"] < season_to_predict - 1]
    train_data = train_data[train_data["IP"] > min_ip]

    valid = [c for c in feature_cols if c in train_data.columns and train_data[c].notna().any()]
    X_train = train_data[valid]
    y_train = train_data["ERA_next"]

    model = xgb.XGBRegressor(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
    )
    model.fit(X_train, y_train)

    target = ps[ps["Season"] == season_to_predict - 1].copy()
    target = target[target["IP"] > min_ip]
    target["ERA_predicted"] = model.predict(target[valid])

    return model, target, valid


# ── Load config and data ──────────────────────────────────────
with open(CONFIGS_DIR / "optimized.json") as f:
    cfg = json.load(f)

base_features = cfg["base_features"]
engineering = cfg.get("engineering", {})

base_df = load_base_data()
ps, all_features = engineer_features(base_df, base_features, engineering)

years = [2023, 2024, 2025]

for year in years:
    print(f"\n{'='*60}")
    print(f"  {year} Season Evaluation")
    print(f"{'='*60}")

    model, predictions, valid_features = train_and_predict(ps, all_features, year, cfg)

    actual = ps[ps["Season"] == year]
    if actual.empty:
        print(f"  No actuals for {year}, skipping")
        continue

    merged = predictions.merge(actual[["IDfg", "ERA"]], on="IDfg", suffixes=("_prev", "_actual"))
    if merged.empty:
        print(f"  No overlap between predictions and actuals for {year}")
        continue

    mae_model = mean_absolute_error(merged["ERA_actual"], merged["ERA_predicted"])
    mae_naive = mean_absolute_error(merged["ERA_actual"], merged["ERA_prev"])
    print(f"  Model MAE: {mae_model:.4f}")
    print(f"  Naive MAE: {mae_naive:.4f}")

    # ── Predicted vs Actual scatter ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(merged["ERA_actual"], merged["ERA_predicted"], alpha=0.6, s=50, color="steelblue", edgecolors="white", linewidth=0.5)
    lims = [max(1.5, merged["ERA_actual"].min() - 0.5), min(8, merged["ERA_actual"].max() + 0.5)]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5)
    ax.set_xlabel("Actual ERA", fontsize=13)
    ax.set_ylabel("Predicted ERA", fontsize=13)
    ax.set_title(f"{year} ERA — Predicted vs Actual", fontsize=15)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{year}_predicted_vs_actual.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── MAE bar chart ─────────────────────────────────────────
    steamer_path = DATA_DIR / f"steamer-{year}.csv"
    zips_path = DATA_DIR / f"zips-{year}.csv"

    labels = ["Model", "Naive (prev yr)"]
    vals = [mae_model, mae_naive]
    colors = ["steelblue", "gray"]

    if steamer_path.exists() and zips_path.exists():
        steamer = pd.read_csv(steamer_path).dropna(subset=["PlayerId", "ERA"])
        zips_df = pd.read_csv(zips_path).dropna(subset=["PlayerId", "ERA"])
        steamer = steamer[steamer["PlayerId"].astype(str).str.isnumeric()]
        zips_df = zips_df[zips_df["PlayerId"].astype(str).str.isnumeric()]
        steamer["PlayerId"] = steamer["PlayerId"].astype(int)
        zips_df["PlayerId"] = zips_df["PlayerId"].astype(int)

        zips_df = zips_df.rename(columns={"ERA": "ERA_zips"})
        steamer = steamer.rename(columns={"ERA": "ERA_steamer"})

        comp = merged.merge(zips_df[["PlayerId", "ERA_zips"]], left_on="IDfg", right_on="PlayerId", how="inner")
        comp = comp.merge(steamer[["PlayerId", "ERA_steamer"]], left_on="IDfg", right_on="PlayerId", how="inner")

        if not comp.empty:
            mae_steamer = mean_absolute_error(comp["ERA_actual"], comp["ERA_steamer"])
            mae_zips = mean_absolute_error(comp["ERA_actual"], comp["ERA_zips"])
            mae_model_comp = mean_absolute_error(comp["ERA_actual"], comp["ERA_predicted"])
            mae_naive_comp = mean_absolute_error(comp["ERA_actual"], comp["ERA_prev"])

            labels = ["Model", "Steamer", "ZiPS", "Naive"]
            vals = [mae_model_comp, mae_steamer, mae_zips, mae_naive_comp]
            colors = ["steelblue", "coral", "forestgreen", "gray"]

            print(f"  vs Steamer MAE: {mae_steamer:.4f}")
            print(f"  vs ZiPS MAE: {mae_zips:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Mean Absolute Error", fontsize=13)
    ax.set_title(f"{year} ERA Projection Accuracy", fontsize=15)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{year}_mae_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── Feature importance ─────────────────────────────────────
    importance = pd.Series(model.feature_importances_, index=valid_features).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    importance.sort_values().plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel("Importance", fontsize=13)
    ax.set_title(f"{year} — Top 20 Feature Importance", fontsize=15)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{year}_feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"  Charts saved to blog_assets/")

# ── Write config summary ──────────────────────────────────────
summary_lines = [
    "# Optimized Model Configuration",
    "",
    "## Model Parameters",
    f"- n_estimators: {cfg['n_estimators']}",
    f"- max_depth: {cfg['max_depth']}",
    f"- learning_rate: {cfg['learning_rate']}",
    f"- min_ip: {cfg['min_ip']}",
    "",
    f"## Base Features ({len(base_features)})",
    "",
]

# Group base features for readability
base_by_type = {}
for f in sorted(base_features):
    if "(sc)" in f:
        base_by_type.setdefault("Statcast", []).append(f)
    elif "(pi)" in f:
        base_by_type.setdefault("PitchInfo", []).append(f)
    elif "+" in f:
        base_by_type.setdefault("Plus Stats", []).append(f)
    elif f.lower().startswith("bot"):
        base_by_type.setdefault("Botball", []).append(f)
    else:
        base_by_type.setdefault("Traditional", []).append(f)

for group, feats in sorted(base_by_type.items()):
    summary_lines.append(f"### {group} ({len(feats)})")
    for f in feats:
        summary_lines.append(f"- {f}")
    summary_lines.append("")

# Engineering
eng_count = sum(len(v) for v in engineering.values())
summary_lines.append(f"## Engineered Features ({eng_count} total)")
summary_lines.append("")
for suffix, cols in engineering.items():
    label = {
        "_t1": "Last year value",
        "_t2": "2 years ago value",
        "_delta_1yr": "1-year delta",
        "_avg_3yr": "3-year average",
    }.get(suffix, suffix)
    summary_lines.append(f"### {label} ({len(cols)} features)")
    for c in sorted(cols):
        summary_lines.append(f"- {c}")
    summary_lines.append("")

with open(OUT_DIR / "model_summary.md", "w") as f:
    f.write("\n".join(summary_lines))

print(f"\nModel summary written to blog_assets/model_summary.md")
print(f"\nAll assets in: {OUT_DIR}")
