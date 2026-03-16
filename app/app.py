import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CONFIGS_DIR = Path(__file__).resolve().parent.parent / "data" / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Metadata columns (never usable as features) ──────────────
EXCLUDE_COLS = {"IDfg", "Season", "Name", "Team", "Age Rng"}

# ── Categorize features for organized display ────────────────
FEATURE_CATEGORIES = {
    "Traditional / General": lambda c: not any(tag in c for tag in ["(sc)", "(pi)", "bot", "Stf+", "Loc+", "Pit+"]) and "+" not in c,
    "Statcast Pitch Data": lambda c: "(sc)" in c,
    "PitchInfo Pitch Data": lambda c: "(pi)" in c,
    "Plus Stats (park-adjusted)": lambda c: "+" in c and "(sc)" not in c and "(pi)" not in c and not any(c.startswith(p) for p in ["Stf+", "Loc+", "Pit+"]),
    "Stuff+ / Location+ / Pitching+": lambda c: any(c.startswith(p) for p in ["Stf+", "Loc+", "Pit+", "Stuff+", "Location+", "Pitching+"]),
    "Botball Models": lambda c: c.lower().startswith("bot"),
}

# ── Lag/engineering options ────────────────────────────────────
ENGINEERING_OPTIONS = {
    "Last year value (_t1)": "_t1",
    "2 years ago value (_t2)": "_t2",
    "1-year delta (change from last year)": "_delta_1yr",
    "3-year average": "_avg_3yr",
}


def categorize_columns(columns: list[str]) -> dict[str, list[str]]:
    """Sort columns into display categories."""
    cats = {name: [] for name in FEATURE_CATEGORIES}
    for col in columns:
        placed = False
        # Check specific categories first (order matters — most specific first)
        for cat_name in ["Statcast Pitch Data", "PitchInfo Pitch Data",
                         "Stuff+ / Location+ / Pitching+", "Botball Models",
                         "Plus Stats (park-adjusted)"]:
            if FEATURE_CATEGORIES[cat_name](col):
                cats[cat_name].append(col)
                placed = True
                break
        if not placed:
            cats["Traditional / General"].append(col)
    # Sort within each category and drop empties
    return {k: sorted(v) for k, v in cats.items() if v}


# ── Config persistence ────────────────────────────────────────

def list_configs() -> list[str]:
    return sorted(p.stem for p in CONFIGS_DIR.glob("*.json"))


def load_config(name: str) -> dict:
    with open(CONFIGS_DIR / f"{name}.json") as f:
        return json.load(f)


def save_config(name: str, cfg: dict):
    with open(CONFIGS_DIR / f"{name}.json", "w") as f:
        json.dump(cfg, f, indent=2)


def delete_config(name: str):
    path = CONFIGS_DIR / f"{name}.json"
    if path.exists():
        path.unlink()


# ── Data loading ──────────────────────────────────────────────

@st.cache_data
def load_base_data():
    """Load the raw pitching stats (before feature engineering)."""
    return pd.read_parquet(DATA_DIR / "pitching_stats.parquet")


@st.cache_data
def get_available_features(df_columns: tuple) -> list[str]:
    """Get all numeric columns usable as features."""
    df = load_base_data()
    numeric = df.select_dtypes(include="number").columns.tolist()
    return sorted(c for c in numeric if c not in EXCLUDE_COLS)


def engineer_features(df: pd.DataFrame, base_features: list[str], engineering: dict[str, list[str]]) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply time-series feature engineering.

    Args:
        df: Base dataframe sorted by player/season
        base_features: Selected raw features
        engineering: Maps suffix -> list of columns to engineer

    Returns:
        (dataframe with new columns, list of all feature column names)
    """
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

    # Add the target column
    ps["ERA_next"] = ps.groupby("IDfg")["ERA"].shift(-1)

    return ps, all_feature_cols


def train_and_predict(ps, feature_cols, season_to_predict, n_estimators, max_depth, learning_rate, min_ip):
    train_data = ps.dropna(subset=["ERA_next"])
    train_data = train_data[train_data["Season"] < season_to_predict - 1]
    train_data = train_data[train_data["IP"] > min_ip]

    valid_features = [c for c in feature_cols if c in train_data.columns and train_data[c].notna().any()]

    X_train = train_data[valid_features]
    y_train = train_data["ERA_next"]

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    model.fit(X_train, y_train)

    target = ps[ps["Season"] == season_to_predict - 1].copy()
    target = target[target["IP"] > min_ip]
    X_pred = target[valid_features]
    target["ERA_predicted"] = model.predict(X_pred)

    return model, target, valid_features


# ─── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="Pitcher Projections", layout="wide")
st.title("MLB Pitcher ERA Projections")
st.caption("XGBoost model trained on Statcast-era data (2015-2025)")

base_df = load_base_data()
available_features = get_available_features(tuple(base_df.columns))
available_seasons = sorted(base_df["Season"].unique())
predict_seasons = [s for s in available_seasons if s >= 2017]
categorized = categorize_columns(available_features)

# ─── Seed default config ──────────────────────────────────────
if not (CONFIGS_DIR / "dylan-config.json").exists():
    with open(DATA_DIR / "feature_columns.json") as f:
        dylan_features = json.load(f)
    # Split into base features and engineered
    dylan_base = [f for f in dylan_features if not any(f.endswith(s) for s in ["_t1", "_t2", "_delta_1yr", "_avg_3yr"])]
    dylan_eng = {}
    for f in dylan_features:
        for suffix in ["_delta_1yr", "_t1", "_t2"]:
            if f.endswith(suffix):
                base = f[: -len(suffix)]
                dylan_eng.setdefault(suffix, [])
                if base not in dylan_eng[suffix]:
                    dylan_eng[suffix].append(base)
    save_config("dylan-config", {
        "base_features": dylan_base,
        "engineering": dylan_eng,
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.07,
        "min_ip": 15,
        "season_to_predict": 2025,
    })

# ─── Sidebar: saved configs ──────────────────────────────────
st.sidebar.header("Saved Configs")

configs = list_configs()
selected_config = st.sidebar.selectbox("Load config", ["(none)"] + configs)

# Defaults
default_base_features = available_features
default_engineering: dict[str, list[str]] = {}
default_n_est = 500
default_depth = 6
default_lr = 0.07
default_min_ip = 15
default_season = predict_seasons[-1]

if selected_config != "(none)":
    cfg = load_config(selected_config)
    # Support both old format (flat "features") and new format ("base_features" + "engineering")
    if "base_features" in cfg:
        default_base_features = [f for f in cfg["base_features"] if f in available_features]
        default_engineering = cfg.get("engineering", {})
    elif "features" in cfg:
        # Old format: split into base vs engineered
        default_base_features = [f for f in cfg["features"] if f in available_features and not any(f.endswith(s) for s in ["_t1", "_t2", "_delta_1yr", "_avg_3yr"])]
        default_engineering = {}
        for f in cfg["features"]:
            for suffix in ["_delta_1yr", "_t1", "_t2", "_avg_3yr"]:
                if f.endswith(suffix):
                    base = f[: -len(suffix)]
                    if base in available_features:
                        default_engineering.setdefault(suffix, [])
                        if base not in default_engineering[suffix]:
                            default_engineering[suffix].append(base)
    default_n_est = cfg.get("n_estimators", 500)
    default_depth = cfg.get("max_depth", 6)
    default_lr = cfg.get("learning_rate", 0.07)
    default_min_ip = cfg.get("min_ip", 15)
    default_season = cfg.get("season_to_predict", predict_seasons[-1])

# ─── Sidebar: model parameters ────────────────────────────────
st.sidebar.header("Model Parameters")
n_estimators = st.sidebar.slider("n_estimators", 50, 1500, default_n_est, step=50)
max_depth = st.sidebar.slider("max_depth", 2, 12, default_depth)
learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.30, default_lr, step=0.01)

st.sidebar.header("Data Filters")
min_ip = st.sidebar.number_input("Minimum IP", min_value=1, max_value=100, value=default_min_ip)
season_idx = predict_seasons.index(default_season) if default_season in predict_seasons else len(predict_seasons) - 1
season_to_predict = st.sidebar.selectbox("Season to predict", options=predict_seasons, index=season_idx)

# ─── Sidebar: save / delete config ────────────────────────────
st.sidebar.header("Save Config")
save_name = st.sidebar.text_input("Config name")

# ─── Main: Feature Selection (two tabs) ───────────────────────
st.header("Feature Selection")

tab_base, tab_eng = st.tabs(["Base Features", "Engineered Features (time-series)"])

# Number of columns for checkbox grids
N_COLS = 4


def _toggle_all(keys: list[str], select_all_key: str):
    """Callback: sync individual checkboxes to the select-all state."""
    val = st.session_state[select_all_key]
    for k in keys:
        st.session_state[k] = val


# ── Tab 1: Base feature selection by category ─────────────────
with tab_base:
    st.caption("Select the raw stats to include as model inputs. Organized by category — expand each to pick individual features.")

    # Initialize session state defaults for all base feature checkboxes (first run only)
    for cat_cols in categorized.values():
        for feat in cat_cols:
            sk = f"base_{feat}"
            if sk not in st.session_state:
                st.session_state[sk] = feat in default_base_features

    selected_base_features: list[str] = []

    for cat_name, cat_cols in categorized.items():
        child_keys = [f"base_{feat}" for feat in cat_cols]
        n_selected = sum(1 for k in child_keys if st.session_state.get(k, False))

        with st.expander(f"{cat_name} — {n_selected}/{len(cat_cols)} selected", expanded=False):
            sa_key = f"all_{cat_name}"
            st.checkbox(
                "Select all",
                value=n_selected == len(cat_cols),
                key=sa_key,
                on_change=_toggle_all,
                args=(child_keys, sa_key),
            )

            cols = st.columns(N_COLS)
            for i, feat in enumerate(cat_cols):
                with cols[i % N_COLS]:
                    if st.checkbox(feat, key=f"base_{feat}"):
                        selected_base_features.append(feat)

    st.info(f"**{len(selected_base_features)}** base features selected out of {len(available_features)} available")

# ── Tab 2: Feature engineering ────────────────────────────────
with tab_eng:
    st.caption(
        "Add time-lagged versions of your selected base features. "
        "These let the model see trends and history — e.g., did a pitcher's velocity drop year-over-year?"
    )

    engineering_selections: dict[str, list[str]] = {}

    eligible_for_engineering = sorted(selected_base_features)

    if not eligible_for_engineering:
        st.warning("Select some base features first.")
    else:
        # Initialize session state defaults for engineering checkboxes (first run only)
        for suffix in ENGINEERING_OPTIONS.values():
            eng_defaults_for_suffix = set(default_engineering.get(suffix, []))
            for feat in eligible_for_engineering:
                sk = f"eng_{suffix}_{feat}"
                if sk not in st.session_state:
                    st.session_state[sk] = feat in eng_defaults_for_suffix

        for label, suffix in ENGINEERING_OPTIONS.items():
            child_keys = [f"eng_{suffix}_{feat}" for feat in eligible_for_engineering]
            n_eng = sum(1 for k in child_keys if st.session_state.get(k, False))

            with st.expander(f"{label} — {n_eng}/{len(eligible_for_engineering)} selected", expanded=False):
                sa_key = f"alleng_{suffix}"
                st.checkbox(
                    "Select all",
                    value=n_eng == len(eligible_for_engineering),
                    key=sa_key,
                    on_change=_toggle_all,
                    args=(child_keys, sa_key),
                )

                chosen = []
                cols = st.columns(N_COLS)
                for i, feat in enumerate(eligible_for_engineering):
                    with cols[i % N_COLS]:
                        if st.checkbox(feat, key=f"eng_{suffix}_{feat}"):
                            chosen.append(feat)
                if chosen:
                    engineering_selections[suffix] = chosen

        total_eng = sum(len(v) for v in engineering_selections.values())
        if total_eng:
            st.info(f"**{total_eng}** engineered features will be created ({len(selected_base_features)} base + {total_eng} engineered = **{len(selected_base_features) + total_eng}** total)")

if not selected_base_features:
    st.error("Select at least one base feature.")
    st.stop()

# ─── Build features and train ─────────────────────────────────
with st.spinner("Engineering features & training model..."):
    processed_df, all_feature_cols = engineer_features(base_df, selected_base_features, engineering_selections)
    model, predictions, valid_features = train_and_predict(
        processed_df, all_feature_cols, season_to_predict, n_estimators, max_depth, learning_rate, min_ip,
    )

# ─── Save config button (after features are resolved) ─────────
def _build_current_config():
    return {
        "base_features": selected_base_features,
        "engineering": engineering_selections,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "min_ip": min_ip,
        "season_to_predict": season_to_predict,
    }

if st.sidebar.button("Save"):
    if save_name.strip():
        save_config(save_name.strip(), _build_current_config())
        st.sidebar.success(f"Saved '{save_name.strip()}'")
        st.rerun()

if selected_config != "(none)":
    if st.sidebar.button(f"Delete '{selected_config}'"):
        delete_config(selected_config)
        st.rerun()

# ─── Predictions table ────────────────────────────────────────
st.header(f"{season_to_predict} ERA Predictions")

display_cols = ["Name", "Team", "Age", "IP", "ERA", "ERA_predicted"]
available_display = [c for c in display_cols if c in predictions.columns]
st.dataframe(
    predictions[available_display].sort_values("ERA_predicted").reset_index(drop=True),
    use_container_width=True,
    height=400,
)

# ─── Evaluation vs actuals (if available) ─────────────────────
actual = processed_df[processed_df["Season"] == season_to_predict]
if not actual.empty:
    merged = predictions.merge(actual[["IDfg", "ERA"]], on="IDfg", suffixes=("_prev", "_actual"))

    st.header("Model Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(merged["ERA_actual"], merged["ERA_predicted"], alpha=0.6, color="steelblue")
        lims = [
            max(1.5, merged["ERA_actual"].min() - 0.5),
            min(8, merged["ERA_actual"].max() + 0.5),
        ]
        ax.plot(lims, lims, "--", color="gray", alpha=0.5)
        ax.set_xlabel("Actual ERA")
        ax.set_ylabel("Predicted ERA")
        ax.set_title("Predicted vs Actual")
        st.pyplot(fig)

    with col2:
        mae = mean_absolute_error(merged["ERA_actual"], merged["ERA_predicted"])
        naive_mae = mean_absolute_error(merged["ERA_actual"], merged["ERA_prev"])
        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(["Model", "Naive (prev yr)"], [mae, naive_mae], color=["steelblue", "gray"])
        ax.set_ylabel("Mean Absolute Error")
        ax.set_title("Projection Accuracy")
        for bar, v in zip(bars, [mae, naive_mae]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
        st.pyplot(fig)

    # Compare with Steamer/ZiPS if CSVs exist
    steamer_path = DATA_DIR / f"steamer-{season_to_predict}.csv"
    zips_path = DATA_DIR / f"zips-{season_to_predict}.csv"

    if steamer_path.exists() and zips_path.exists():
        steamer = pd.read_csv(steamer_path).dropna(subset=["PlayerId", "ERA"])
        zips_df = pd.read_csv(zips_path).dropna(subset=["PlayerId", "ERA"])
        steamer = steamer[steamer["PlayerId"].astype(str).str.isnumeric()]
        zips_df = zips_df[zips_df["PlayerId"].astype(str).str.isnumeric()]
        steamer["PlayerId"] = steamer["PlayerId"].astype(int)
        zips_df["PlayerId"] = zips_df["PlayerId"].astype(int)

        zips_df = zips_df.rename(columns={"ERA": "ERA_zips"})
        steamer = steamer.rename(columns={"ERA": "ERA_steamer"})

        comp = merged.merge(zips_df[["PlayerId", "ERA_zips"]], left_on="IDfg", right_on="PlayerId")
        comp = comp.merge(steamer[["PlayerId", "ERA_steamer"]], left_on="IDfg", right_on="PlayerId")

        if not comp.empty:
            st.subheader("vs Steamer & ZiPS")
            mae_model = mean_absolute_error(comp["ERA_actual"], comp["ERA_predicted"])
            mae_steamer = mean_absolute_error(comp["ERA_actual"], comp["ERA_steamer"])
            mae_zips = mean_absolute_error(comp["ERA_actual"], comp["ERA_zips"])
            mae_naive = mean_absolute_error(comp["ERA_actual"], comp["ERA_prev"])

            fig, ax = plt.subplots(figsize=(7, 5))
            labels = ["Model", "Steamer", "ZiPS", "Naive"]
            vals = [mae_model, mae_steamer, mae_zips, mae_naive]
            colors = ["steelblue", "coral", "forestgreen", "gray"]
            bars = ax.bar(labels, vals, color=colors)
            ax.set_ylabel("Mean Absolute Error")
            ax.set_title(f"{season_to_predict} ERA — MAE Comparison")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
            st.pyplot(fig)

# ─── Feature importance ───────────────────────────────────────
st.header("Feature Importance (top 20)")
importance = pd.Series(model.feature_importances_, index=valid_features).sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(8, 6))
importance.sort_values().plot.barh(ax=ax, color="steelblue")
ax.set_xlabel("Importance")
st.pyplot(fig)
