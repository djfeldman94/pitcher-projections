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
        for cat_name in ["Statcast Pitch Data", "PitchInfo Pitch Data",
                         "Stuff+ / Location+ / Pitching+", "Botball Models",
                         "Plus Stats (park-adjusted)"]:
            if FEATURE_CATEGORIES[cat_name](col):
                cats[cat_name].append(col)
                placed = True
                break
        if not placed:
            cats["Traditional / General"].append(col)
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
    return pd.read_parquet(DATA_DIR / "pitching_stats.parquet")


@st.cache_data
def get_available_features(df_columns: tuple) -> list[str]:
    df = load_base_data()
    numeric = df.select_dtypes(include="number").columns.tolist()
    return sorted(c for c in numeric if c not in EXCLUDE_COLS)


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


# ── Helpers: collect current widget state into a config dict ──

def _gather_current_selections():
    """Read all widget values and return the full pending config."""
    base_feats = []
    for cat_cols in categorized.values():
        for feat in cat_cols:
            if st.session_state.get(f"base_{feat}", False):
                base_feats.append(feat)

    eng = {}
    for suffix in ENGINEERING_OPTIONS.values():
        chosen = []
        for feat in sorted(base_feats):
            if st.session_state.get(f"eng_{suffix}_{feat}", False):
                chosen.append(feat)
        if chosen:
            eng[suffix] = chosen

    return {
        "base_features": base_feats,
        "engineering": eng,
        "n_estimators": st.session_state.get("w_n_estimators", 500),
        "max_depth": st.session_state.get("w_max_depth", 6),
        "learning_rate": st.session_state.get("w_learning_rate", 0.07),
        "min_ip": st.session_state.get("w_min_ip", 15),
        "season_to_predict": st.session_state.get("w_season", 2025),
    }


def _configs_differ(a: dict, b: dict) -> bool:
    """Check if two configs differ in any meaningful way."""
    for key in ["n_estimators", "max_depth", "min_ip", "season_to_predict"]:
        if a.get(key) != b.get(key):
            return True
    if abs(a.get("learning_rate", 0) - b.get("learning_rate", 0)) > 1e-6:
        return True
    if sorted(a.get("base_features", [])) != sorted(b.get("base_features", [])):
        return True
    a_eng = a.get("engineering", {})
    b_eng = b.get("engineering", {})
    if set(a_eng.keys()) != set(b_eng.keys()):
        return True
    for k in a_eng:
        if sorted(a_eng[k]) != sorted(b_eng.get(k, [])):
            return True
    return False


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
    dylan_base = [f for f in dylan_features if not any(f.endswith(s) for s in ["_t1", "_t2", "_delta_1yr", "_avg_3yr"])]
    dylan_eng = {}
    for feat in dylan_features:
        for suffix in ["_delta_1yr", "_t1", "_t2"]:
            if feat.endswith(suffix):
                base = feat[: -len(suffix)]
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


def _parse_config(cfg: dict) -> tuple[list[str], dict[str, list[str]]]:
    """Parse a saved config into (base_features, engineering) regardless of format."""
    if "base_features" in cfg:
        base = [f for f in cfg["base_features"] if f in available_features]
        eng = cfg.get("engineering", {})
    elif "features" in cfg:
        base = [f for f in cfg["features"] if f in available_features and not any(f.endswith(s) for s in ["_t1", "_t2", "_delta_1yr", "_avg_3yr"])]
        eng = {}
        for feat in cfg["features"]:
            for suffix in ["_delta_1yr", "_t1", "_t2", "_avg_3yr"]:
                if feat.endswith(suffix):
                    b = feat[: -len(suffix)]
                    if b in available_features:
                        eng.setdefault(suffix, [])
                        if b not in eng[suffix]:
                            eng[suffix].append(b)
    else:
        base, eng = [], {}
    return base, eng


def _push_config_to_widgets(cfg: dict):
    """Write a saved config into all widget session state keys."""
    base_feats, engineering = _parse_config(cfg)
    base_set = set(base_feats)

    # Set all base feature checkboxes
    for cat_cols in categorized.values():
        for feat in cat_cols:
            st.session_state[f"base_{feat}"] = feat in base_set

    # Set all engineering checkboxes
    for suffix in ENGINEERING_OPTIONS.values():
        eng_set = set(engineering.get(suffix, []))
        for feat in available_features:
            sk = f"eng_{suffix}_{feat}"
            if sk in st.session_state:
                st.session_state[sk] = feat in eng_set

    # Set model parameter widgets
    st.session_state["w_n_estimators"] = cfg.get("n_estimators", 500)
    st.session_state["w_max_depth"] = cfg.get("max_depth", 6)
    st.session_state["w_learning_rate"] = cfg.get("learning_rate", 0.07)
    st.session_state["w_min_ip"] = cfg.get("min_ip", 15)
    season = cfg.get("season_to_predict", predict_seasons[-1])
    if season in predict_seasons:
        st.session_state["w_season"] = season


# Detect config dropdown change and push values into widget state
if selected_config != "(none)":
    if st.session_state.get("_loaded_config_name") != selected_config:
        cfg = load_config(selected_config)
        _push_config_to_widgets(cfg)
        st.session_state["_loaded_config_name"] = selected_config
        st.rerun()

    cfg = load_config(selected_config)
    default_base_features, default_engineering = _parse_config(cfg)
    default_n_est = cfg.get("n_estimators", 500)
    default_depth = cfg.get("max_depth", 6)
    default_lr = cfg.get("learning_rate", 0.07)
    default_min_ip = cfg.get("min_ip", 15)
    default_season = cfg.get("season_to_predict", predict_seasons[-1])
else:
    if st.session_state.get("_loaded_config_name") is not None:
        st.session_state["_loaded_config_name"] = None

# ─── Sidebar: model parameters ────────────────────────────────
st.sidebar.header("Model Parameters")
n_estimators = st.sidebar.slider("n_estimators", 50, 1500, default_n_est, step=50, key="w_n_estimators")
max_depth = st.sidebar.slider("max_depth", 2, 12, default_depth, key="w_max_depth")
learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.30, default_lr, step=0.01, key="w_learning_rate")

st.sidebar.header("Data Filters")
min_ip = st.sidebar.number_input("Minimum IP", min_value=1, max_value=100, value=default_min_ip, key="w_min_ip")
season_idx = predict_seasons.index(default_season) if default_season in predict_seasons else len(predict_seasons) - 1
season_to_predict = st.sidebar.selectbox("Season to predict", options=predict_seasons, index=season_idx, key="w_season")

# ─── Sidebar: save / delete config ────────────────────────────
st.sidebar.header("Save Config")
save_name = st.sidebar.text_input("Config name")

# ─── Main: Feature Selection (two tabs) ───────────────────────
st.header("Feature Selection")

tab_base, tab_eng = st.tabs(["Base Features", "Engineered Features (time-series)"])

N_COLS = 4


def _toggle_all(keys: list[str], select_all_key: str):
    val = st.session_state[select_all_key]
    for k in keys:
        st.session_state[k] = val


def _clear_keys(keys: list[str]):
    for k in keys:
        st.session_state[k] = False


# ── Tab 1: Base feature selection by category ─────────────────
with tab_base:
    st.caption("Select the raw stats to include as model inputs. Organized by category — expand each to pick individual features.")

    for cat_cols in categorized.values():
        for feat in cat_cols:
            sk = f"base_{feat}"
            if sk not in st.session_state:
                st.session_state[sk] = feat in default_base_features

    all_base_keys = [f"base_{feat}" for cat_cols in categorized.values() for feat in cat_cols]
    st.button("Clear all base features", on_click=_clear_keys, args=(all_base_keys,), key="reset_base")

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
        for suffix in ENGINEERING_OPTIONS.values():
            eng_defaults_for_suffix = set(default_engineering.get(suffix, []))
            for feat in eligible_for_engineering:
                sk = f"eng_{suffix}_{feat}"
                if sk not in st.session_state:
                    st.session_state[sk] = feat in eng_defaults_for_suffix

        all_eng_keys = [f"eng_{suffix}_{feat}" for suffix in ENGINEERING_OPTIONS.values() for feat in eligible_for_engineering]
        st.button("Clear all engineered features", on_click=_clear_keys, args=(all_eng_keys,), key="reset_eng")

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

# ─── Apply button / change detection ─────────────────────────
pending_config = _gather_current_selections()

# On first run, auto-apply so there's something to show
if "applied_config" not in st.session_state:
    st.session_state["applied_config"] = pending_config

has_changes = _configs_differ(pending_config, st.session_state["applied_config"])

if has_changes:
    st.markdown("---")
    col_apply, col_msg = st.columns([1, 3])
    with col_apply:
        if st.button("Apply Changes", type="primary", use_container_width=True):
            st.session_state["applied_config"] = pending_config
            st.rerun()
    with col_msg:
        st.warning("You have unapplied changes. Click **Apply Changes** to retrain the model.", icon="\u26a0\ufe0f")
    st.markdown("---")

# ─── Train using the APPLIED config (cached in session state) ─
applied = st.session_state["applied_config"]

# Only retrain when the applied config actually changes
applied_key = json.dumps(applied, sort_keys=True, default=str)
if st.session_state.get("_trained_key") != applied_key:
    with st.spinner("Engineering features & training model..."):
        processed_df, all_feature_cols = engineer_features(
            base_df, applied["base_features"], applied["engineering"],
        )
        model, predictions, valid_features = train_and_predict(
            processed_df, all_feature_cols,
            applied["season_to_predict"],
            applied["n_estimators"],
            applied["max_depth"],
            applied["learning_rate"],
            applied["min_ip"],
        )
        st.session_state["_trained_key"] = applied_key
        st.session_state["_trained_results"] = (model, predictions, valid_features, processed_df)
else:
    model, predictions, valid_features, processed_df = st.session_state["_trained_results"]

# ─── Save config button (uses pending so you save what you see) ─
if st.sidebar.button("Save"):
    if save_name.strip():
        save_config(save_name.strip(), pending_config)
        st.sidebar.success(f"Saved '{save_name.strip()}'")
        st.rerun()

if selected_config != "(none)":
    if st.sidebar.button(f"Delete '{selected_config}'"):
        delete_config(selected_config)
        st.rerun()

# ─── Predictions table ────────────────────────────────────────
st.header(f"{applied['season_to_predict']} ERA Predictions")

display_cols = ["Name", "Team", "Age", "IP", "ERA", "ERA_predicted"]
available_display = [c for c in display_cols if c in predictions.columns]
st.dataframe(
    predictions[available_display].sort_values("ERA_predicted").reset_index(drop=True),
    use_container_width=True,
    height=400,
)

# ─── Evaluation vs actuals (if available) ─────────────────────
actual = processed_df[processed_df["Season"] == applied["season_to_predict"]]
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

    steamer_path = DATA_DIR / f"steamer-{applied['season_to_predict']}.csv"
    zips_path = DATA_DIR / f"zips-{applied['season_to_predict']}.csv"

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
            ax.set_title(f"{applied['season_to_predict']} ERA — MAE Comparison")
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
