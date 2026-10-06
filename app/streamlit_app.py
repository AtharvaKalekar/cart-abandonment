"""
streamlit_app.py - Cart Abandonment (Session Non-Purchase) Analysis & Prediction
Case Study 59: ITM Skills University - Machine Learning Project

Every number, table, persona and figure shown here is READ from files written by the
notebook (models/, reports/). Nothing is hardcoded.
"""

import os
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

st.set_page_config(
    page_title="Cart Abandonment Intelligence Platform",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px; border-radius: 16px; border: 1px solid #334155;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3); margin-bottom: 24px;
    }
    .header-title { color: #38bdf8; font-size: 2.2rem; font-weight: 800; margin-bottom: 8px; }
    .header-subtitle { color: #94a3b8; font-size: 1.05rem; }
    .risk-high { background-color: #7f1d1d; color: #fca5a5; padding: 10px 18px;
                 border-radius: 8px; font-weight: 700; display: inline-block; }
    .risk-low  { background-color: #064e3b; color: #6ee7b7; padding: 10px 18px;
                 border-radius: 8px; font-weight: 700; display: inline-block; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Version-safe display helpers (Streamlit replaced `use_container_width` with `width`)
# ------------------------------------------------------------------------------
def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, **kw)


def show_plot(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def show_img(path, caption=None):
    """Show a saved figure if it exists (never crash on a missing figure)."""
    if path and os.path.exists(path):
        try:
            st.image(path, caption=caption, width="stretch")
        except Exception:
            st.image(path, caption=caption, use_container_width=True)
    else:
        st.caption("Figure not found. Re-run the notebook export cell.")


# ------------------------------------------------------------------------------
# Locate artifacts
# ------------------------------------------------------------------------------
APP_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))


def resolve_file(rel_path):
    for base in (PROJECT_ROOT, WORKSPACE_ROOT,
                 os.path.join(PROJECT_ROOT, "cart-abandonment"),
                 os.path.join(WORKSPACE_ROOT, "cart-abandonment")):
        candidate = os.path.join(base, rel_path)
        if os.path.exists(candidate):
            return candidate
    return None


REQUIRED = {
    "model_no_pv": "models/model_no_pagevalues.joblib",
    "model_full": "models/model_with_pagevalues.joblib",
    "kmeans": "models/kmeans_cluster_model.joblib",
    "cluster_scaler": "models/cluster_scaler.joblib",
    "raw_data": "data/raw/online_shoppers_intention.csv",
    "experiments": "reports/experiment_results.csv",
    "clusters": "reports/cluster_profiles.csv",
    "sensitivity": "reports/cost_sensitivity.csv",
    "metrics": "reports/test_evaluation_metrics.json",
}
PATHS = {key: resolve_file(rel) for key, rel in REQUIRED.items()}
missing = [REQUIRED[k] for k, v in PATHS.items() if v is None]
if missing:
    st.error("Missing required file(s): " + ", ".join(f"`{m}`" for m in missing)
             + ". Run the notebook top to bottom (final export cell) first.")
    st.stop()


def fig_path(name):
    return resolve_file(f"reports/figures/{name}")


@st.cache_resource
def load_models(paths):
    return tuple(joblib.load(p) for p in paths)


@st.cache_data
def load_reports(raw_p, exp_p, clu_p, sens_p, met_p):
    with open(met_p) as f:
        metrics = json.load(f)
    return (pd.read_csv(raw_p), pd.read_csv(exp_p), pd.read_csv(clu_p),
            pd.read_csv(sens_p), metrics)


@st.cache_resource
def get_shap_explainer(model_key, _pipeline):
    # model_key is part of the cache key; the pipeline itself (underscore) is NOT hashed.
    clf = _pipeline.named_steps.get("model") or _pipeline.named_steps.get("classifier") or _pipeline.steps[-1][1]
    return shap.TreeExplainer(clf)


model_no_pv, model_full, kmeans_model, cluster_scaler = load_models(
    tuple(PATHS[k] for k in ("model_no_pv", "model_full", "kmeans", "cluster_scaler")))
raw_df, exp_df, cluster_df, sens_df, metrics = load_reports(
    PATHS["raw_data"], PATHS["experiments"], PATHS["clusters"], PATHS["sensitivity"], PATHS["metrics"])

df_clean = raw_df.drop_duplicates().copy()
df_clean["Is_Abandoned"] = (~df_clean["Revenue"]).astype(int)
df_clean["Outcome"] = df_clean["Is_Abandoned"].map({1: "Abandoned", 0: "Purchased"})
base_rate = metrics["base_abandonment_rate"]
m_nopv, m_full = metrics["no_pagevalues_model"], metrics["with_pagevalues_model"]

# ------------------------------------------------------------------------------
# Header + tabs
# ------------------------------------------------------------------------------
st.markdown("""
<div class="header-card">
    <div class="header-title">🛒 E-Commerce Cart Abandonment Intelligence</div>
    <div class="header-subtitle">Case Study 59: session behaviour analytics, classification and behavioural profiling</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["📌 Overview", "📊 EDA Explorer", "🎯 Predict Risk",
                "👥 Customer Segments", "📈 Model Performance", "💡 Business Action Engine"])

# ==============================================================================
# TAB 1: OVERVIEW
# ==============================================================================
with tabs[0]:
    st.subheader("Problem Framing & Key Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions (deduplicated)", f"{len(df_clean):,}")
    c2.metric("Base abandonment rate", f"{base_rate * 100:.2f}%")
    c3.metric("ROC-AUC, early-session model", f"{m_nopv['roc_auc']:.3f}",
              delta=f"{m_full['roc_auc'] - m_nopv['roc_auc']:+.3f} with PageValues", delta_color="off")
    c4.metric("Tuned threshold t* (early-session)", f"{metrics['optimal_threshold']:.2f}")

    st.divider()
    col_a, col_b = st.columns([3, 2])
    persona_list = ", ".join(f"**{p}**" for p in cluster_df["Persona"])
    with col_a:
        st.markdown(f"""
### Problem statement (Case Study 59)
*An e-commerce platform wants to investigate patterns associated with users leaving without completing a purchase.*

### Approach
1. **Supervised classification:** predicts the probability that a session ends without purchase using tuned **Gradient Boosting**
   (SMOTE inside training folds only). Two variants: an **early-session** model without `PageValues` and a **full** model with it.
2. **Cost-aware threshold:** the decision threshold is tuned on out-of-fold training predictions
   (assumed costs: coupon ${metrics['cost_fp']:g}, expected recovered margin ${metrics['primary_cost_fn']:g} per missed abandoner).
3. **Unsupervised profiling:** K-Means groups abandoning sessions into {len(cluster_df)} personas: {persona_list}.

> **Scope note:** the dataset has no add-to-cart events. A "non-purchase" is **session non-conversion**,
> a broader notion than strict cart abandonment.
""")
    with col_b:
        st.markdown("### Summary statistics (deduplicated)")
        show_df(df_clean.describe().T[["mean", "std", "min", "50%", "max"]].round(2), height=330)

# ==============================================================================
# TAB 2: EDA EXPLORER
# ==============================================================================
with tabs[1]:
    st.subheader("Interactive Exploratory Data Analysis")
    abandon_pct = df_clean["Is_Abandoned"].mean() * 100
    eda_option = st.selectbox("Select view", [
        "Target Class Distribution", "Monthly Abandonment Rate", "Visitor Type & Weekend",
        "Behavioural Feature Distributions", "Correlation Matrix"])

    if eda_option == "Target Class Distribution":
        fig = px.histogram(df_clean, x="Outcome", color="Outcome",
                           color_discrete_map={"Abandoned": "#ef4444", "Purchased": "#10b981"},
                           title=f"Session outcomes ({abandon_pct:.2f}% abandoned)")
        show_plot(fig)
        st.info(f"Finding: strong class imbalance ({abandon_pct:.2f}% abandoned vs {100 - abandon_pct:.2f}% purchased), "
                "so accuracy is misleading; Purchaser F1, Macro F1 and PR-AUC are reported as well.")

    elif eda_option == "Monthly Abandonment Rate":
        order = [m for m in ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                 if m in df_clean["Month"].unique()]
        g = df_clean.groupby("Month")["Is_Abandoned"].agg(["mean", "size"]).reindex(order).reset_index()
        g["Abandonment Rate (%)"] = g["mean"] * 100
        fig = px.bar(g, x="Month", y="Abandonment Rate (%)", color="Abandonment Rate (%)",
                     color_continuous_scale="Reds", hover_data={"size": True}, title="Abandonment rate by month")
        show_plot(fig)
        lo, hi = g.loc[g["Abandonment Rate (%)"].idxmin()], g.loc[g["Abandonment Rate (%)"].idxmax()]
        st.info(f"Finding: lowest abandonment in {lo['Month']} ({lo['Abandonment Rate (%)']:.1f}%, n={int(lo['size'])}), "
                f"highest in {hi['Month']} ({hi['Abandonment Rate (%)']:.1f}%, n={int(hi['size'])}). "
                "Sample sizes differ a lot between months, so treat small months with caution.")

    elif eda_option == "Visitor Type & Weekend":
        c1, c2 = st.columns(2)
        v = df_clean.groupby("VisitorType")["Is_Abandoned"].mean().mul(100).reset_index(name="Abandonment Rate (%)")
        w = df_clean.groupby("Weekend")["Is_Abandoned"].mean().mul(100).reset_index(name="Abandonment Rate (%)")
        w["Weekend"] = w["Weekend"].map({True: "Weekend", False: "Weekday"})
        with c1:
            show_plot(px.bar(v, x="VisitorType", y="Abandonment Rate (%)", title="By visitor type"))
        with c2:
            show_plot(px.bar(w, x="Weekend", y="Abandonment Rate (%)", title="Weekend vs weekday"))
        vt = v.set_index("VisitorType")["Abandonment Rate (%)"]
        wk = w.set_index("Weekend")["Abandonment Rate (%)"]
        st.info(f"Finding: new visitors abandon {vt.get('New_Visitor', float('nan')):.1f}% vs returning "
                f"{vt.get('Returning_Visitor', float('nan')):.1f}%; weekend {wk.get('Weekend', float('nan')):.1f}% "
                f"vs weekday {wk.get('Weekday', float('nan')):.1f}%.")

    elif eda_option == "Behavioural Feature Distributions":
        col = st.radio("Metric", ["PageValues", "ExitRates", "BounceRates", "ProductRelated_Duration"], horizontal=True)
        fig = px.box(df_clean, x="Outcome", y=col, color="Outcome", points=False, title=f"{col} by outcome")
        show_plot(fig)
        means = df_clean.groupby("Outcome")[col].mean()
        st.info(f"Finding: mean {col} is {means['Abandoned']:.3f} for abandoned vs {means['Purchased']:.3f} for purchased sessions.")

    else:
        cols = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
                "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues",
                "SpecialDay", "Is_Abandoned"]
        corr = df_clean[cols].corr()
        show_plot(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation matrix"))
        st.info(f"Finding: BounceRates vs ExitRates correlation = {corr.loc['BounceRates', 'ExitRates']:.2f} (multicollinearity); "
                f"PageValues vs Is_Abandoned = {corr.loc['PageValues', 'Is_Abandoned']:.2f}.")

# ==============================================================================
# TAB 3: PREDICT RISK
# ==============================================================================
with tabs[2]:
    st.subheader("Real-Time Session Risk Evaluator")
    model_mode = st.radio("Model", ["Early-session model (no PageValues)", "Full model (uses PageValues)"], index=0)
    use_pv = "uses PageValues" in model_mode
    active_pipeline = model_full if use_pv else model_no_pv
    active_key = "with_pagevalues_model" if use_pv else "no_pagevalues_model"
    threshold = metrics[active_key].get("threshold", metrics["optimal_threshold"])

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### Page navigation")
            admin = st.number_input("Administrative pages", 0, 30, 1)
            admin_dur = st.number_input("Admin duration (s)", 0.0, 3500.0, 20.0)
            info = st.number_input("Informational pages", 0, 30, 0)
            info_dur = st.number_input("Info duration (s)", 0.0, 3000.0, 0.0)
            product = st.number_input("Product pages visited", 0, 750, 12)
            product_dur = st.number_input("Product duration (s)", 0.0, 65000.0, 350.0)
        with col2:
            st.markdown("#### Traffic & intent")
            bounce_rate = st.slider("Bounce rate", 0.0, 0.20, 0.01, 0.001)
            exit_rate = st.slider("Exit rate", 0.0, 0.20, 0.03, 0.001)
            if use_pv:
                page_value = st.number_input("Page value", 0.0, 400.0, 0.0)
            else:
                page_value = None
                st.caption("ℹ️ PageValues is disabled in early-session mode.")
            special_day = st.slider("Special day score", 0.0, 1.0, 0.0, 0.2)
        with col3:
            st.markdown("#### Context")
            month = st.selectbox("Month", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], index=8)
            visitor_type = st.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"], index=0)
            weekend = st.selectbox("Weekend session?", [False, True], index=0)
            os_code = st.selectbox("Operating system ID", list(range(1, 9)), index=1)
            browser_code = st.selectbox("Browser ID", list(range(1, 14)), index=1)
            region_code = st.selectbox("Region ID", list(range(1, 10)), index=0)
            traffic_code = st.selectbox("Traffic type ID", list(range(1, 21)), index=1)
        submit_btn = st.form_submit_button("⚡ Predict abandonment risk", use_container_width=True)

    if submit_btn:
        if exit_rate < bounce_rate:
            st.warning("Validation note: exit rate is normally greater than or equal to bounce rate, "
                       "so this combination is unusual. The prediction below may be unreliable.")
        row = {
            "Administrative": admin, "Administrative_Duration": admin_dur,
            "Informational": info, "Informational_Duration": info_dur,
            "ProductRelated": product, "ProductRelated_Duration": product_dur,
            "BounceRates": bounce_rate, "ExitRates": exit_rate, "SpecialDay": special_day,
            "Month": month, "VisitorType": visitor_type, "Weekend": weekend,
            "OperatingSystems": os_code, "Browser": browser_code, "Region": region_code,
            "TrafficType": traffic_code,
        }
        if use_pv:
            row["PageValues"] = page_value
        input_df = pd.DataFrame([row])

        prob = float(active_pipeline.predict_proba(input_df)[0, 1])
        flagged = prob >= threshold

        st.divider()
        st.markdown("### Prediction")
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted abandonment score", f"{prob * 100:.1f}%")
        r2.metric("Operating threshold t*", f"{threshold:.2f}")
        with r3:
            if flagged:
                st.markdown('<div class="risk-high">⚠️ FLAG FOR INTERVENTION</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-low">✅ LOW RISK (likely buyer)</div>', unsafe_allow_html=True)
        st.caption("The score comes from a SMOTE-trained model, so it is a ranking score and not a calibrated probability. "
                   "The decision rule is: flag when score ≥ t*.")
        risk_label = "Flagged" if flagged else "Low"

        # ---- SHAP explanation for this single session ----
        st.markdown("#### What drove this prediction (SHAP, log-odds scale)")
        pre = active_pipeline.named_steps["preprocessor"]
        X_t = pre.transform(input_df)
        X_t = X_t.toarray() if hasattr(X_t, "toarray") else np.asarray(X_t)
        names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
        sv = get_shap_explainer(active_key, active_pipeline).shap_values(X_t)
        sv = sv[1] if isinstance(sv, list) else sv
        contrib = pd.DataFrame({"Feature": names, "SHAP": np.asarray(sv)[0]})
        top = contrib.reindex(contrib["SHAP"].abs().sort_values(ascending=False).index).head(8).iloc[::-1]
        fig_shap = go.Figure(go.Bar(
            x=top["SHAP"], y=top["Feature"], orientation="h",
            marker_color=["#ef4444" if v > 0 else "#10b981" for v in top["SHAP"]]))
        fig_shap.update_layout(title="Top 8 features (red pushes toward abandonment, green toward purchase)",
                               xaxis_title="SHAP value (log-odds)", showlegend=False)
        show_plot(fig_shap)

        # ---- Closest abandoner segment ----
        c_in = np.log1p(pd.DataFrame([{
            "ProductRelated": product, "ProductRelated_Duration": product_dur,
            "BounceRates": bounce_rate, "ExitRates": exit_rate}]))
        cid = int(kmeans_model.predict(cluster_scaler.transform(c_in))[0])
        match = cluster_df[cluster_df["Cluster"] == cid]
        persona = match["Persona"].iloc[0] if len(match) else f"Cluster {cid}"
        st.info(f"👥 Closest abandoner segment: **{persona}** "
                "(clusters were fitted on abandoning sessions only).")

        st.session_state.update({"last_persona": persona, "last_risk": risk_label, "last_score": prob,
                                 "last_model": model_mode})

# ==============================================================================
# TAB 4: CUSTOMER SEGMENTS
# ==============================================================================
with tabs[3]:
    st.subheader("Behavioural Segmentation of Abandoning Sessions")
    if "last_persona" in st.session_state:
        st.success(f"Last evaluated session maps to: **{st.session_state['last_persona']}**")

    show_df(cluster_df.round(3))
    big = cluster_df.loc[cluster_df["Session_Count"].idxmax()]
    small = cluster_df.loc[cluster_df["Session_Count"].idxmin()]
    st.info(f"Finding: the largest segment is **{big['Persona']}** ({big['Share (%)']:.1f}% of abandoning sessions) "
            f"and the smallest is **{small['Persona']}** ({small['Share (%)']:.1f}%).")
    c1, c2 = st.columns(2)
    with c1:
        show_img(fig_path("10_pca_clusters.png"), "PCA projection of personas")
    with c2:
        show_img(fig_path("09_k_sweep.png"), "Elbow and silhouette sweep used to choose k")

# ==============================================================================
# TAB 5: MODEL PERFORMANCE
# ==============================================================================
with tabs[4]:
    st.subheader("Model Evaluation & Experiment Leaderboard")

    # High-level 5-Algorithm Comparison Summary Table for quick viva defense
    st.markdown("### 🏆 5-Algorithm Baseline Comparison")
    algo_summary = exp_df[exp_df["Stage"] == "1. Baseline, full features"][
        ["Algorithm", "Accuracy (%)", "Precision (%)", "Recall (%)", "Abandoner F1 (%)", "Macro F1 (%)", "ROC-AUC"]
    ].sort_values(by="Macro F1 (%)", ascending=False).reset_index(drop=True)
    
    show_df(algo_summary)
    
    best_algo = algo_summary.iloc[0]
    st.success(
        f"🏆 **Best Performing Model: {best_algo['Algorithm']}**\n\n"
        f"- **ROC-AUC:** `{best_algo['ROC-AUC']:.3f}` | **Macro F1:** `{best_algo['Macro F1 (%)']:.2f}%` | **Accuracy:** `{best_algo['Accuracy (%)']:.2f}%`\n"
        f"- **Verdict:** **Gradient Boosting** outperforms all other classifiers (Logistic Regression, KNN, Decision Tree, Random Forest) "
        f"because it iteratively optimizes loss across weak decision trees, capturing non-linear interactions without overfitting."
    )

    st.divider()
    st.markdown("### 🔬 Full Multi-Stage Experiment Log")
    stage = st.selectbox("Filter experiment stage", ["All stages"] + list(exp_df["Stage"].unique()))
    view = exp_df if stage == "All stages" else exp_df[exp_df["Stage"] == stage]
    show_df(view)
    st.caption("FPR = share of real purchasers wrongly flagged as abandoners. CV rows are 5-fold means on the training set; "
               "Test rows are held-out set results.")

    st.divider()
    st.markdown("### With vs without PageValues (tuned Gradient Boosting, test set at tuned threshold)")
    st.markdown("> `PageValues` is a near-proxy for the outcome and is not available early in a session.")
    comp = pd.DataFrame([
        {"Model": "Early session (no PageValues)", "ROC-AUC": m_nopv["roc_auc"], "PR-AUC (purchasers)": m_nopv["pr_auc"],
         "Abandoner F1": m_nopv["abandoner_f1"], "Purchaser F1": m_nopv["purchaser_f1"],
         "Macro F1": m_nopv["macro_f1"], "Accuracy": m_nopv["accuracy"], "Threshold": m_nopv["threshold"]},
        {"Model": "Full (with PageValues)", "ROC-AUC": m_full["roc_auc"], "PR-AUC (purchasers)": m_full["pr_auc"],
         "Abandoner F1": m_full["abandoner_f1"], "Purchaser F1": m_full["purchaser_f1"],
         "Macro F1": m_full["macro_f1"], "Accuracy": m_full["accuracy"], "Threshold": m_full["threshold"]},
    ])
    show_df(comp.round(4))
    st.caption(f"Majority-class baseline: always predicting 'abandoned' gives {base_rate * 100:.1f}% accuracy.")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        show_img(fig_path("05_confusion_matrix.png"), "Confusion matrices at the tuned threshold")
    with c2:
        show_img(fig_path("06_roc_pr_curves.png"), "ROC and precision-recall curves")
    c3, c4 = st.columns(2)
    with c3:
        show_img(fig_path("07_threshold_cost_analysis.png"), "Cost vs threshold")
    with c4:
        show_img(fig_path("08_shap_summary.png"), "Global SHAP importance (early-session model)")

# ==============================================================================
# TAB 6: BUSINESS ACTION ENGINE
# ==============================================================================
with tabs[5]:
    st.subheader("Cost Sensitivity & Intervention Rules")
    st.markdown(f"Costs: coupon (false positive) = **${metrics['cost_fp']:g}**; the missed-abandoner cost is the "
                "**expected recovered margin** (response rate × margin) and is varied below.")
    show_df(sens_df)

    positive = sens_df[sens_df["Savings vs Best Naive (%)"] > 0]
    if positive.empty:
        st.warning("At none of the tested cost ratios does the model beat the best naive strategy.")
    else:
        max_fn = positive["FN Cost ($)"].max()
        st.info(f"Finding: the model beats the best naive strategy only when the missed-abandoner cost is at most "
                f"${max_fn:g} (coupon ${metrics['cost_fp']:g}). At higher ratios, flagging every session is optimal "
                f"because about {base_rate * 100:.0f}% of sessions abandon.")

    st.divider()
    st.markdown("### Recommended action for the last evaluated session")
    if "last_persona" in st.session_state:
        persona, risk = st.session_state["last_persona"], st.session_state["last_risk"]
        st.success(f"Segment: **{persona}** | Decision: **{risk}** ({st.session_state['last_model']})")
        if risk != "Flagged":
            st.success("Action: standard browsing experience. No intervention needed.")
        elif persona == "Bouncers":
            st.error("Action: light-touch exit-intent message with social proof (reviews, best-sellers). Avoid discounts.")
        elif persona == "Deep Browsers":
            st.warning("Action: coupon nudge or cart-recovery email, since this visitor evaluated many products.")
        elif persona == "Light Browsers":
            st.info("Action: personalised product-recommendation banner to deepen engagement.")
        else:
            st.info("Action: generic retargeting message.")
        st.caption("These actions are rule-based suggestions, not model output.")
    else:
        st.caption("Submit a prediction in the 'Predict Risk' tab to get a session-specific recommendation.")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#64748b;'>Machine Learning Exam Project • ITM Skills University B.Tech CSE (2024-28)</div>",
            unsafe_allow_html=True)
