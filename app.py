import streamlit as st
import pandas as pd
import plotly.express as px

# ===================== PAGE CONFIG =====================

st.set_page_config(
    page_title="Project Review Dashboard",
    layout="wide"
)

st.title("📊 Project Review Dashboard")

st.markdown(
    """
Interactive dashboard for analysing project reviews from two reviewers  
and supporting funding / risk decisions.

🔒 The Excel file you upload is used only in this session and is not stored.
"""
)

# ===================== FILE UPLOAD =====================

uploaded_file = st.file_uploader("📁 Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file is None:
    st.info("Upload the Excel file that contains your review data.")
    st.stop()

df = pd.read_excel(uploaded_file)

# ===================== COLUMN ALIASES / CLEANUP =====================

# Main project name
if "Research project title:" in df.columns:
    df["Project_Name"] = df["Research project title:"].astype(str)
elif "Project title2" in df.columns:
    df["Project_Name"] = df["Project title2"].astype(str)
else:
    df["Project_Name"] = df.iloc[:, 0].astype(str)

# ID
if "ID" in df.columns:
    df["Project_ID"] = df["ID"]

# Funding band
if "Please specify which of the funding bands is requested for the project?" in df.columns:
    df["Funding_Band"] = df["Please specify which of the funding bands is requested for the project?"]

# Final total (numeric)
if "Final Total" in df.columns:
    df["Final_Total"] = df["Final Total"]

# ===================== DERIVED SCORE COLUMNS =====================

# (label, reviewer1_column, reviewer2_column)
score_pairs = [
    ("Methods",    "Methods_46_review1",    "Methods_46_review2"),
    ("Impact",     "Impact_47_review1",     "Impact_47_review2"),
    ("Innovation", "Innovation_48_review1", "Innovation_48_review2"),
    ("Plan",       "Plan_49_review1",       "Plan_49_review2"),
    ("Team",       "Team_50_review1",       "Team_50_review2"),
    ("Total",      "Total_51_review1",      "Total_51_review2"),
]

for label, c1, c2 in score_pairs:
    if c1 in df.columns and c2 in df.columns:
        df[f"{label}_avg"] = df[[c1, c2]].mean(axis=1)

# ===================== SIDEBAR FILTERS =====================

st.sidebar.header("🔍 Filters")

# Risk category filter
if "Risk_Category" in df.columns:
    risk_options = ["All"] + sorted(df["Risk_Category"].dropna().unique().tolist())
    risk_filter = st.sidebar.selectbox("Risk category", risk_options, index=0)
else:
    risk_filter = "All"

# Funding band filter
if "Funding_Band" in df.columns:
    band_options = ["All"] + sorted(df["Funding_Band"].dropna().unique().tolist())
    band_filter = st.sidebar.selectbox("Funding band", band_options, index=0)
else:
    band_filter = "All"

# Alert filter (e.g., "alert 1")
if "alert" in df.columns:
    alert_options = ["All"] + sorted(df["alert"].dropna().unique().tolist())
    alert_filter = st.sidebar.selectbox("Alert flag", alert_options, index=0)
else:
    alert_filter = "All"

# Minimum final score slider
if "Final_Total" in df.columns and df["Final_Total"].notna().any():
    min_score = float(df["Final_Total"].min())
    max_score = float(df["Final_Total"].max())
    min_final_filter = st.sidebar.slider(
        "Minimum Final Total",
        min_value=min_score,
        max_value=max_score,
        value=min_score,
        step=0.1,
    )
else:
    min_final_filter = None

# ---- Apply filters ----
filtered_df = df.copy()

if risk_filter != "All" and "Risk_Category" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Risk_Category"] == risk_filter]

if band_filter != "All" and "Funding_Band" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Funding_Band"] == band_filter]

if alert_filter != "All" and "alert" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["alert"] == alert_filter]

if min_final_filter is not None and "Final_Total" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Final_Total"] >= min_final_filter]

if filtered_df.empty:
    st.warning("No projects left after applying filters.")
    st.stop()

# ===================== TOP METRICS =====================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Number of projects", len(filtered_df))

with col2:
    if "Final_Total" in filtered_df.columns:
        st.metric("Average Final Total", f"{filtered_df['Final_Total'].mean():.1f}")
    else:
        st.metric("Average Final Total", "N/A")

with col3:
    if "Risk_Category" in filtered_df.columns:
        high_risk = (
            filtered_df["Risk_Category"].fillna("").str.contains("High", case=False).sum()
        )
        st.metric("High-risk projects", int(high_risk))
    else:
        st.metric("High-risk projects", "N/A")

with col4:
    if "Funding_Band" in filtered_df.columns:
        st.metric("Funding bands used", int(filtered_df["Funding_Band"].nunique()))
    else:
        st.metric("Funding bands used", "N/A")

st.markdown("---")

# ===================== TABS =====================

tab_overview, tab_scores, tab_agreement, tab_risk = st.tabs(
    ["📁 Overview", "📈 Score distribution", "⚖️ Reviewer agreement", "🚥 Risk & sentiment"]
)

# ---------- TAB 1: OVERVIEW ----------
with tab_overview:
    st.subheader("Project overview (after filters)")

    cols_to_show = [c for c in [
        "Project_ID",
        "Project_Name",
        "Final_Total",
        "Methods_avg",
        "Impact_avg",
        "Innovation_avg",
        "Plan_avg",
        "Team_avg",
        "Risk_Category",
        "Funding_Band",
        "Project duration",
        "alert",
    ] if c in filtered_df.columns]

    if "Final_Total" in cols_to_show:
        view_df = filtered_df[cols_to_show].sort_values("Final_Total", ascending=False)
    else:
        view_df = filtered_df[cols_to_show]

    st.dataframe(view_df, use_container_width=True)

# ---------- TAB 2: SCORE DISTRIBUTION ----------
with tab_scores:
    st.subheader("Score distribution across projects")

    score_options = []
    if "Final_Total" in filtered_df.columns:
        score_options.append("Final_Total")
    for label, _, _ in score_pairs:
        col_name = f"{label}_avg"
        if col_name in filtered_df.columns:
            score_options.append(col_name)

    if not score_options:
        st.info("No numeric score columns found.")
    else:
        selected_score = st.selectbox("Select score to analyse", score_options)

        fig_hist = px.histogram(
            filtered_df,
            x=selected_score,
            nbins=15,
            title=f"Distribution of {selected_score}",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        if "Project_Name" in filtered_df.columns:
            fig_bar = px.bar(
                filtered_df.sort_values(by=selected_score, ascending=False),
                x="Project_Name",
                y=selected_score,
                color="Risk_Category" if "Risk_Category" in filtered_df.columns else None,
                hover_data=["Funding_Band"] if "Funding_Band" in filtered_df.columns else None,
                title=f"{selected_score} by project",
            )
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)

# ---------- TAB 3: REVIEWER AGREEMENT ----------
with tab_agreement:
    st.subheader("Reviewer agreement")

    # Only dimensions where both reviewer columns exist
    dims_dict = {
        label: (c1, c2)
        for (label, c1, c2) in score_pairs
        if c1 in df.columns and c2 in df.columns
    }

    if not dims_dict:
        st.info("No reviewer 1 & 2 score pairs found.")
    else:
        dim = st.selectbox("Dimension", list(dims_dict.keys()))
        c1, c2 = dims_dict[dim]

        temp = filtered_df[["Project_Name", c1, c2]].dropna()
        temp["abs_diff"] = (temp[c1] - temp[c2]).abs()

        if temp.empty:
            st.info("No data for this dimension after filters.")
        else:
            fig_diff = px.bar(
                temp.sort_values("abs_diff", ascending=False),
                x="Project_Name",
                y="abs_diff",
                hover_data=[c1, c2],
                title=f"Reviewer disagreement on {dim}",
            )
            fig_diff.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_diff, use_container_width=True)

            threshold = st.slider(
                "Highlight projects with absolute difference ≥",
                float(temp["abs_diff"].min()),
                float(temp["abs_diff"].max()),
                value=2.0,
                step=0.5,
            )

            st.write("Projects above threshold:")
            st.dataframe(temp[temp["abs_diff"] >= threshold])

# ---------- TAB 4: RISK & SENTIMENT ----------
with tab_risk:
    st.subheader("Risk, sentiment and decisions")

    if {"Final_Total", "Sentiment_Polarity"}.issubset(filtered_df.columns):
        fig_scatter = px.scatter(
            filtered_df,
            x="Final_Total",
            y="Sentiment_Polarity",
            color="Risk_Category" if "Risk_Category" in filtered_df.columns else None,
            hover_data=["Project_Name", "Funding_Band"]
            if "Funding_Band" in filtered_df.columns
            else ["Project_Name"],
            title="Final score vs sentiment polarity",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Need both 'Final Total' and 'Sentiment_Polarity' to draw the scatter plot.")

    st.markdown("### Inspect a single project")

    if "Project_Name" in filtered_df.columns:
        project_list = filtered_df["Project_Name"].dropna().unique().tolist()
        selected_project = st.selectbox("Select project", project_list)

        proj_rows = filtered_df[filtered_df["Project_Name"] == selected_project]
        row = proj_rows.iloc[0]

        st.markdown(f"**Project ID:** {row.get('Project_ID', 'N/A')}")
        st.markdown(f"**Risk category:** {row.get('Risk_Category', 'N/A')}")
        st.markdown(f"**Funding band:** {row.get('Funding_Band', 'N/A')}")
        st.markdown(f"**Final total:** {row.get('Final_Total', 'N/A')}")
        st.markdown(f"**Recommendation comparison score:** {row.get('Recommendation_Comparison_Score', 'N/A')}")

        if "Combined_Comments" in row:
            with st.expander("Combined comments"):
                st.write(row["Combined_Comments"])

        if "Risk_Explanation" in row:
            with st.expander("Risk explanation"):
                st.write(row["Risk_Explanation"])

        score_cols = [c for c in [
            "Methods_46_review1", "Methods_46_review2", "Methods_avg",
            "Impact_47_review1", "Impact_47_review2", "Impact_avg",
            "Innovation_48_review1", "Innovation_48_review2", "Innovation_avg",
            "Plan_49_review1", "Plan_49_review2", "Plan_avg",
            "Team_50_review1", "Team_50_review2", "Team_avg",
            "Total_51_review1", "Total_51_review2", "Total_avg",
            "Final_Total",
        ] if c in proj_rows.columns]

        if score_cols:
            st.markdown("**Score details**")
            st.dataframe(
                proj_rows[score_cols].T.rename(columns={proj_rows.index[0]: "Score"})
            )
    else:
        st.info("No projects to display.")
