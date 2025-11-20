import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re

# ===================== PAGE CONFIG =====================

st.set_page_config(
    page_title="Project Review Dashboard",
    layout="wide"
)

st.title("📊 Project Review Dashboard")

st.markdown(
    """
Interactive dashboard for analysing project reviews, reviewer agreement, risk and comments.

🔒 The Excel file you upload is used only in this session and is not stored.
"""
)

# ===================== FILE UPLOAD =====================

uploaded_file = st.file_uploader("📁 Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file is None:
    st.info("Upload the Excel file that contains your review data.")
    st.stop()

df = pd.read_excel(uploaded_file)

# ===================== COLUMN ALIASES / CLEANUP (NEW FILE) =====================

def first_nonnull(df, candidates):
    """Return a Series that is the first non-null across the listed columns (left to right)."""
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        return pd.Series([np.nan] * len(df), index=df.index)
    return df[cols].bfill(axis=1).iloc[:, 0]

# Project title
df["Project_Name"] = first_nonnull(
    df,
    [
        "Project_title2_ref",
        "Project_title_4_review1",
        "Project_title_4_review2",
    ],
).astype(str)

# Project ID
df["Project_ID"] = first_nonnull(
    df,
    ["ID3_Key", "ID_0_review1", "ID_0_review2"],
).astype(str)

# Funding band (coalesce r1/r2)
df["Funding_Band"] = first_nonnull(
    df,
    ["Band_47_review1", "Band_47_review2"],
)

# Budget total from both reviewers (optional)
df["Budget_Total"] = first_nonnull(
    df,
    ["Budget_Total_46_review1", "Budget_Total_46_review2"],
)

# Budget: use review2 as main budget, numeric, euros
if "Budget_Total_46_review2" in df.columns:
    df["Budget_Total_46_review2"] = pd.to_numeric(df["Budget_Total_46_review2"], errors="coerce")
    df["Budget_EUR"] = df["Budget_Total_46_review2"]
else:
    df["Budget_EUR"] = np.nan

# Duration -> keep the name your overview tab expects ("Project duration")
df["Project duration"] = first_nonnull(
    df,
    ["Duration_49_review1", "Duration_49_review2"],
)

# Category / Multi-sport (optional display fields)
df["Category"] = first_nonnull(df, ["Category_51_review1", "Category_51_review2"])
df["Multi_Sport"] = first_nonnull(df, ["Multi_Sport_50_review1", "Multi_Sport_50_review2"])

# Group from Category_51_review1 (or fallback)
if "Category_51_review1" in df.columns:
    df["Group"] = df["Category_51_review1"]
elif "Category_51_review2" in df.columns:
    df["Group"] = df["Category_51_review2"]
else:
    df["Group"] = df["Category"]

# Alias for Final_Total (your code expects this name)
if "Final Total" in df.columns and "Final_Total" not in df.columns:
    df["Final_Total"] = df["Final Total"]

# Risk fields – not in this file; if you later add a risk column, map it to "Risk_Category"
# e.g. df["Risk_Category"] = df["Some_Risk_Column"]

# Sentiment fields – not present in the new file; keep None so risk tab hides charts
sentiment_num_col = None
sentiment_label_col = None

# ===================== DERIVED SCORE COLUMNS (AVERAGE OF REVIEWERS) =====================

score_pairs = [
    ("Methods",    "Methods_53_review1",    "Methods_53_review2"),
    ("Impact",     "Impact_54_review1",     "Impact_54_review2"),
    ("Innovation", "Innovation_55_review1", "Innovation_55_review2"),
    ("Plan",       "Plan_56_review1",       "Plan_56_review2"),
    ("Team",       "Team_57_review1",       "Team_57_review2"),
    ("Total",      "Total_58_review1",      "Total_58_review2"),
]

for label, c1, c2 in score_pairs:
    if c1 in df.columns and c2 in df.columns:
        df[f"{label}_avg"] = df[[c1, c2]].mean(axis=1)

# ===================== SIDEBAR FILTERS =====================

st.sidebar.header("🔍 Filters")

# Risk filter
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

# Alert filter (if present)
if "alert" in df.columns:
    alert_options = ["All"] + sorted(df["alert"].dropna().unique().tolist())
    alert_filter = st.sidebar.selectbox("Alert flag", alert_options, index=0)
else:
    alert_filter = "All"

# Minimum final score (uses alias Final_Total)
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

# ===================== BIG DIFFERENCE & MAX DIFF =====================

big_diff_cols = [c for c in [
    "diff_Methods", "diff_Impact", "diff_Innovation", "diff_Plan", "diff_Team"
] if c in filtered_df.columns]

if big_diff_cols:
    filtered_df["Any_big_diff"] = (filtered_df[big_diff_cols].abs() >= 4).any(axis=1)
    filtered_df["Max_diff"] = filtered_df[big_diff_cols].abs().max(axis=1)
else:
    filtered_df["Any_big_diff"] = False
    filtered_df["Max_diff"] = 0

# ===================== BUCKET HELPER =====================

def infer_bucket(row) -> str:
    """
    Default bucket assignment based on Multi_Sport & Group/Category text.
    """
    multi = str(row.get("Multi_Sport", "")).strip().lower()
    group = str(row.get("Group", "")).strip().lower()
    cat = str(row.get("Category", "")).strip().lower()

    is_multi = multi in ["yes", "y", "true", "1", "multi", "multi-sport"]
    text = f"{group} {cat}"

    is_paralympic = "paralympic" in text
    is_para = "para" in text  # catches para-sport / para sport etc.

    if is_paralympic and is_multi:
        return "1 - Priority multi-sport Paralympic"
    elif is_paralympic:
        return "2 - Priority one-sport Paralympic"
    elif is_para:
        return "3 - Other para sports"
    else:
        return "4 - Others"  # bucket 5 is manual reject

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
    if "Budget_EUR" in filtered_df.columns and filtered_df["Budget_EUR"].notna().any():
        st.metric("Total budget (filtered)", f"€{filtered_df['Budget_EUR'].sum():,.0f}")
    else:
        st.metric("Total budget (filtered)", "N/A")

with col4:
    if "Funding_Band" in filtered_df.columns:
        st.metric("Funding bands used", int(filtered_df["Funding_Band"].nunique()))
    else:
        st.metric("Funding bands used", "N/A")

st.markdown("---")

# ===================== TABS =====================

tab_overview, tab_scores, tab_agreement, tab_risk, tab_profiles, tab_comments, tab_decision, tab_buckets, tab_bucket_bands, tab_dragdrop, tab_buckets4 = st.tabs(
    [
        "📁 Overview",
        "📈 Scores & funding",
        "⚖️ Reviewer agreement",
        "🚥 Risk & sentiment",
        "🧬 Project profiles",
        "🗯️ Comment insights",
        "🧠 Decision support",
        "🏷️ Buckets & prioritization",
        "💶 Buckets by funding band",
        "🖱️ Drag & drop buckets",
        "🏷️ Buckets (4 only)",
    ]
)



# ---------- TAB 1: OVERVIEW ----------
with tab_overview:
    st.subheader("Project overview (after filters)")

    cols_to_show = [c for c in [
        "Project_ID", "Project_Name", "Final_Total",
        "Methods_avg", "Impact_avg", "Innovation_avg", "Plan_avg", "Team_avg",
        "Budget_EUR",
        "Risk_Category", "Funding_Band", "Project duration", "alert",
    ] if c in filtered_df.columns]

    if "Final_Total" in cols_to_show:
        view_df = filtered_df[cols_to_show].sort_values("Final_Total", ascending=False)
    else:
        view_df = filtered_df[cols_to_show]

    if "Budget_EUR" in view_df.columns:
        styled = view_df.style.format({"Budget_EUR": "€{:,.0f}".format})
        st.dataframe(styled, use_container_width=True)
    else:
        st.dataframe(view_df, use_container_width=True)

# ---------- TAB 2: SCORES & FUNDING ----------
with tab_scores:
    st.subheader("Score distributions")

    # Histogram & bar by project
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

        fig_hist = px.histogram(filtered_df, x=selected_score, nbins=15)
        st.plotly_chart(fig_hist, use_container_width=True)

        if "Project_Name" in filtered_df.columns:
            hover_cols = []
            if "Funding_Band" in filtered_df.columns:
                hover_cols.append("Funding_Band")
            if "Budget_EUR" in filtered_df.columns:
                hover_cols.append("Budget_EUR")

            fig_bar = px.bar(
                filtered_df.sort_values(by=selected_score, ascending=False),
                x="Project_Name",
                y=selected_score,
                color="Risk_Category" if "Risk_Category" in filtered_df.columns else None,
                hover_data=hover_cols or None,
                title=f"{selected_score} by project",
            )
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### Funding efficiency")

    # Box plot: Final_Total by Funding_Band
    if "Final_Total" in filtered_df.columns and "Funding_Band" in filtered_df.columns:
        fig_box = px.box(
            filtered_df,
            x="Funding_Band",
            y="Final_Total",
            color="Risk_Category" if "Risk_Category" in filtered_df.columns else None,
            title="Distribution of Final Scores by Funding Band",
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Need 'Final_Total' and 'Funding_Band' for funding efficiency analysis.")

    st.markdown("### Budget distribution")
    if "Budget_EUR" in filtered_df.columns and filtered_df["Budget_EUR"].notna().any():
        fig_budget = px.histogram(
            filtered_df,
            x="Budget_EUR",
            nbins=15,
            title="Budget (EUR) distribution"
        )
        st.plotly_chart(fig_budget, use_container_width=True)
    else:
        st.info("No budget data available.")

    st.markdown("### Final score vs risk (bubble chart)")
    if "Final_Total" in filtered_df.columns and "Risk_Category" in filtered_df.columns:
        fig_bubble = px.scatter(
            filtered_df,
            x="Final_Total",
            y="Risk_Category",
            size="Innovation_avg" if "Innovation_avg" in filtered_df.columns else None,
            color="Risk_Category",
            hover_data=[
                col for col in ["Project_ID", "Project_Name", "Funding_Band", "Budget_EUR"]
                if col in filtered_df.columns
            ],
            title="Final score vs risk (bubble size = Innovation score)",
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

# ---------- TAB 3: REVIEWER AGREEMENT ----------
with tab_agreement:
    st.subheader("Reviewer agreement (projects with ≥ 4-point difference in any item)")

    # Only projects that have at least one big diff in Methods/Impact/Innovation/Plan/Team
    big_df = filtered_df[filtered_df["Any_big_diff"]] if "Any_big_diff" in filtered_df.columns else pd.DataFrame()

    if big_df.empty:
        st.info("No projects with ≥ 4-point difference in any of Methods / Impact / Innovation / Plan / Team.")
    else:
        # Dimension -> (reviewer1_col, reviewer2_col, diff_col)
        dim_info = {
            "Methods":    ("Methods_53_review1",    "Methods_53_review2",    "diff_Methods"),
            "Impact":     ("Impact_54_review1",     "Impact_54_review2",     "diff_Impact"),
            "Innovation": ("Innovation_55_review1", "Innovation_55_review2", "diff_Innovation"),
            "Plan":       ("Plan_56_review1",       "Plan_56_review2",       "diff_Plan"),
            "Team":       ("Team_57_review1",       "Team_57_review2",       "diff_Team"),
        }

        dim_info = {
            name: (c1, c2, cdiff)
            for name, (c1, c2, cdiff) in dim_info.items()
            if c1 in big_df.columns and c2 in big_df.columns and cdiff in big_df.columns
        }

        if not dim_info:
            st.info("Reviewer score columns not found for Methods/Impact/Innovation/Plan/Team.")
        else:
            dim = st.selectbox("Dimension", list(dim_info.keys()))
            c1, c2, cdiff = dim_info[dim]

            cols = ["Project_ID", "Project_Name", c1, c2, cdiff]
            if "Budget_EUR" in big_df.columns:
                cols.append("Budget_EUR")

            temp = big_df[cols].dropna()
            temp["abs_diff"] = temp[cdiff].abs()
            temp = temp[temp["abs_diff"] >= 4]

            if temp.empty:
                st.info(f"No projects with ≥ 4-point difference in {dim}.")
            else:
                hover_cols = ["Project_ID", c1, c2]
                if "Budget_EUR" in temp.columns:
                    hover_cols.append("Budget_EUR")

                fig_diff = px.bar(
                    temp.sort_values("abs_diff", ascending=False),
                    x="Project_Name",
                    y="abs_diff",
                    hover_data=hover_cols,
                    title=f"Reviewer disagreement on {dim} (diff ≥ 4)",
                )
                fig_diff.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_diff, use_container_width=True)

                st.markdown("**Projects with ≥ 4-point difference in this dimension:**")
                show_cols = ["Project_ID", "Project_Name", c1, c2, "abs_diff"]
                if "Budget_EUR" in temp.columns:
                    show_cols.append("Budget_EUR")

                st.dataframe(
                    temp[show_cols].sort_values("abs_diff", ascending=False),
                    use_container_width=True,
                )

    st.markdown("### Average differences by criterion")
    if big_diff_cols:
        mean_diffs = filtered_df[big_diff_cols].abs().mean()
        heat_df = pd.DataFrame({"Dimension": mean_diffs.index, "Mean_abs_diff": mean_diffs.values})
        fig_heat = px.imshow(
            [heat_df["Mean_abs_diff"].values],
            x=heat_df["Dimension"],
            y=["Avg diff"],
            color_continuous_scale="Reds",
            aspect="auto",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("### Reviewer 1 vs Reviewer 2 total score")
    if "Total_58_review1" in filtered_df.columns and "Total_58_review2" in filtered_df.columns:
        fig_r12 = px.scatter(
            filtered_df,
            x="Total_58_review1",
            y="Total_58_review2",
            hover_data=[
                col for col in ["Project_ID", "Project_Name", "Budget_EUR"]
                if col in filtered_df.columns
            ],
            title="Reviewer 1 vs Reviewer 2 total scores",
        )
        fig_r12.add_shape(
            type="line",
            x0=filtered_df["Total_58_review1"].min(),
            y0=filtered_df["Total_58_review1"].min(),
            x1=filtered_df["Total_58_review1"].max(),
            y1=filtered_df["Total_58_review1"].max(),
        )
        st.plotly_chart(fig_r12, use_container_width=True)

# ---------- TAB 4: RISK & SENTIMENT ----------
with tab_risk:
    st.subheader("Risk & sentiment overview")

    cols = [c for c in [
        "Project_ID", "Project_Name",
        "Risk_Category",
        sentiment_label_col,
        sentiment_num_col,
        "Final_Total",
        "Budget_EUR",
    ] if c is not None and c in filtered_df.columns]

    if cols:
        st.markdown("**Projects with risk and sentiment:**")
        st.dataframe(filtered_df[cols], use_container_width=True)
    else:
        st.info("No risk/sentiment columns available.")

    # Scatter: Final_Total vs sentiment (not used now, no sentiment fields)
    if sentiment_num_col and sentiment_num_col in filtered_df.columns and "Final_Total" in filtered_df.columns:
        fig_scatter = px.scatter(
            filtered_df,
            x="Final_Total",
            y=sentiment_num_col,
            color="Risk_Category" if "Risk_Category" in filtered_df.columns else None,
            hover_data=["Project_ID", "Project_Name"],
            title=f"Final_Total vs {sentiment_num_col}",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### Inspect single project")
    if "Project_Name" in filtered_df.columns:
        project_list = filtered_df["Project_Name"].dropna().unique().tolist()
        selected_project = st.selectbox("Select project", project_list)

        proj_rows = filtered_df[filtered_df["Project_Name"] == selected_project]
        row = proj_rows.iloc[0]

        st.markdown(f"**Project ID:** {row.get('Project_ID', 'N/A')}")
        st.markdown(f"**Project name:** {row.get('Project_Name', 'N/A')}")
        st.markdown(f"**Risk:** {row.get('Risk_Category', 'N/A')}")
        st.markdown(f"**Final total:** {row.get('Final_Total', 'N/A')}")
        st.markdown(f"**Budget (EUR):** {row.get('Budget_EUR', 'N/A')}")


        if sentiment_label_col:
            st.markdown(f"**Sentiment label:** {row.get(sentiment_label_col, 'N/A')}")
        if sentiment_num_col:
            st.markdown(f"**Sentiment score:** {row.get(sentiment_num_col, 'N/A')}")

        if "Combined_Comments" in row:
            with st.expander("Combined comments"):
                st.write(row["Combined_Comments"])
        if "Combined_Overall_Feedback" in row:
            with st.expander("Combined overall feedback"):
                st.write(row["Combined_Overall_Feedback"])
        if "All_Feedback_Text" in row:
            with st.expander("All feedback text"):
                st.write(row["All_Feedback_Text"])
        if "Risk_Explanation" in row:
            with st.expander("Risk explanation"):
                st.write(row["Risk_Explanation"])

# ---------- TAB 5: PROJECT PROFILES ----------
with tab_profiles:
    st.subheader("Per-project profiles & reviewer comparison")

    # === 5.1 Radar chart of average scores ===
    radar_cols = [c for c in ["Methods_avg", "Impact_avg", "Innovation_avg", "Plan_avg", "Team_avg"] if c in filtered_df.columns]
    if radar_cols and "Project_Name" in filtered_df.columns:
        project_list = filtered_df["Project_Name"].dropna().unique().tolist()
        selected_project = st.selectbox("Select project", project_list, key="radar_project")

        proj_rows = filtered_df[filtered_df["Project_Name"] == selected_project]
        row = proj_rows.iloc[0]

        values = [row[c] for c in radar_cols]
        categories = radar_cols

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=selected_project
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title="Project performance radar (average of reviewers)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown(f"**Project ID:** {row.get('Project_ID', 'N/A')}")
        st.markdown(f"**Project name:** {row.get('Project_Name', 'N/A')}")
        st.markdown(f"**Final total:** {row.get('Final_Total', 'N/A')}")
        st.markdown(f"**Budget (EUR):** {row.get('Budget_EUR', 'N/A')}")


        # === 5.2 Reviewer comparison per criterion (bar chart) ===
        st.markdown("### Reviewer 1 vs Reviewer 2 per criterion")

        dim_info = [
            ("Methods",    "Methods_53_review1",    "Methods_53_review2"),
            ("Impact",     "Impact_54_review1",     "Impact_54_review2"),
            ("Innovation", "Innovation_55_review1", "Innovation_55_review2"),
            ("Plan",       "Plan_56_review1",       "Plan_56_review2"),
            ("Team",       "Team_57_review1",       "Team_57_review2"),
            ("Total",      "Total_58_review1",      "Total_58_review2"),
        ]

        rows_comp = []
        diff_rows = []
        for label, c1, c2 in dim_info:
            if c1 in proj_rows.columns and c2 in proj_rows.columns:
                s1 = row[c1]
                s2 = row[c2]
                if pd.notna(s1) or pd.notna(s2):
                    rows_comp.append({"Criterion": label, "Reviewer": "Reviewer 1", "Score": s1})
                    rows_comp.append({"Criterion": label, "Reviewer": "Reviewer 2", "Score": s2})
                    diff_rows.append({
                        "Criterion": label,
                        "Reviewer 1": s1,
                        "Reviewer 2": s2,
                        "Diff (|R1 - R2|)": abs(s1 - s2) if pd.notna(s1) and pd.notna(s2) else None
                    })

        if rows_comp:
            comp_df = pd.DataFrame(rows_comp)
            fig_comp = px.bar(
                comp_df,
                x="Criterion",
                y="Score",
                color="Reviewer",
                barmode="group",
                title="Reviewer 1 vs Reviewer 2 scores for this project",
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            st.markdown("**Score differences per criterion**")
            diff_df = pd.DataFrame(diff_rows)
            st.dataframe(diff_df, use_container_width=True)
        else:
            st.info("No reviewer 1/2 score columns found for this project.")

    else:
        st.info("Need average scores for Methods/Impact/Innovation/Plan/Team to draw project profiles.")

    # === 5.3 Reviewer disagreement vs final score (all projects) ===
    st.markdown("### Reviewer disagreement vs final score (all projects)")
    if "Max_diff" in filtered_df.columns and "Final_Total" in filtered_df.columns:
        fig_md = px.scatter(
            filtered_df,
            x="Max_diff",
            y="Final_Total",
            color="Risk_Category" if "Risk_Category" in filtered_df.columns else None,
            hover_data=[
                col for col in ["Project_ID", "Project_Name", "Budget_EUR"]
                if col in filtered_df.columns
            ],
            title="Max reviewer disagreement vs Final Total",
        )
        st.plotly_chart(fig_md, use_container_width=True)
    else:
        st.info("Need Max_diff and Final_Total to show disagreement vs score.")

# ---------- TAB 6: COMMENT INSIGHTS ----------
with tab_comments:
    st.subheader("🗯️ Comment insights (word cloud & keywords)")

    # Identify comment/feedback/text columns
    comment_columns = []
    for col in df.columns:
        lc = col.lower()
        if any(k in lc for k in ["comment", "feedback", "text"]):
            comment_columns.append(col)

    for col in ["Combined_Comments", "Combined_Overall_Feedback", "All_Feedback_Text"]:
        if col in df.columns and col not in comment_columns:
            comment_columns.append(col)

    if not comment_columns:
        st.info("No comment/feedback/text columns found in the file.")
    else:
        col_choice = st.selectbox("Select text column", comment_columns)

        # Project selection: "All projects" or specific project (with ID)
        project_options = ["All projects"]
        if "Project_Name" in filtered_df.columns:
            if "Budget_EUR" in filtered_df.columns:
                base = filtered_df[["Project_Name", "Project_ID", "Budget_EUR"]]
            else:
                base = filtered_df[["Project_Name", "Project_ID"]]

            base = base.dropna(subset=["Project_Name", "Project_ID"]).drop_duplicates()

            def project_label(row):
                name = row["Project_Name"]
                pid = row["Project_ID"]
                if "Budget_EUR" in row and pd.notna(row.get("Budget_EUR", np.nan)):
                    return f"{name} (ID: {pid}, €{row['Budget_EUR']:,.0f})"
                return f"{name} (ID: {pid})"

            project_options += base.apply(project_label, axis=1).tolist()

        proj_choice = st.selectbox("Select project", project_options)

        subset = filtered_df
        if (
            proj_choice != "All projects"
            and "Project_Name" in filtered_df.columns
            and "Project_ID" in filtered_df.columns
        ):
            try:
                # extract ID between "ID:" and either comma or closing parenthesis
                after_id = proj_choice.split("ID:")[1].strip()
                if "," in after_id:
                    proj_id_str = after_id.split(",")[0].strip()
                else:
                    proj_id_str = after_id.rstrip(")").strip()
                subset = filtered_df[filtered_df["Project_ID"].astype(str) == proj_id_str]
            except Exception:
                subset = filtered_df[filtered_df["Project_Name"] == proj_choice]

        all_text = " ".join(subset[col_choice].dropna().astype(str))

        if not all_text.strip():
            st.warning("No text in this column for the selected project(s).")
        else:
            # Word cloud
            wordcloud = WordCloud(
                width=1200, height=600, background_color="white", collocations=False
            ).generate(all_text)

            st.markdown("**Word cloud**")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            # Simple keyword frequency bar chart
            st.markdown("**Top keywords (frequency)**")
            tokens = re.findall(r"\b\w+\b", all_text.lower())
            stopwords = {
                "the", "and", "for", "with", "that", "this", "are", "was", "were",
                "but", "not", "have", "has", "had", "their", "they", "them",
                "into", "from", "about", "there", "been", "will", "would",
                "could", "should", "can", "may", "might", "such", "also",
                "project", "study", "studies", "research"
            }
            words = [w for w in tokens if w not in stopwords and len(w) > 2]
            if words:
                freq = Counter(words).most_common(20)
                freq_df = pd.DataFrame(freq, columns=["word", "count"])
                fig_kw = px.bar(freq_df, x="word", y="count", title="Top 20 words in selected comments")
                st.plotly_chart(fig_kw, use_container_width=True)
            else:
                st.info("No significant keywords to display.")

            if proj_choice == "All projects":
                st.caption("Text aggregated across all filtered projects.")
            else:
                st.caption(f"Text for: {proj_choice}")

# ---------- TAB 7: DECISION SUPPORT ----------
# ---------- TAB 7: DECISION SUPPORT ----------
with tab_decision:
    st.subheader("🧠 Decision-support ranking (0–1 normalized weights)")

    # Base dataframe for decision scoring
    dec_df = filtered_df.copy()

    # Merge bucket + band info from session bucket_df
    if "bucket_df" in st.session_state:
        bdf = st.session_state["bucket_df"].copy()
        bdf["Project_ID"] = bdf["Project_ID"].astype(str)
        dec_df["Project_ID"] = dec_df["Project_ID"].astype(str)

        keep_cols = ["Project_ID"]
        for c in ["Bucket", "Funding_Band"]:
            if c in bdf.columns:
                keep_cols.append(c)

        meta = bdf[keep_cols].drop_duplicates("Project_ID")
        dec_df = dec_df.merge(meta, on="Project_ID", how="left")

    # Fallback if bucket missing
    if "Bucket" not in dec_df.columns:
        dec_df["Bucket"] = dec_df.apply(infer_bucket, axis=1)

    st.markdown("### Global weight sliders (0 → 1)")

    w_score  = st.slider("Weight of Final Score",  0.0, 1.0, 0.6, 0.05)
    w_bucket = st.slider("Weight of Bucket",       0.0, 1.0, 0.3, 0.05)
    w_band   = st.slider("Weight of Funding Band", 0.0, 1.0, 0.1, 0.05)

    st.markdown("---")
    st.markdown("### Bucket priority weights (0 → 1)")

    default_bucket_order = [
        "1 - Priority multi-sport Paralympic",
        "2 - Priority one-sport Paralympic",
        "3 - Other para sports",
        "4 - Others",
        "5 - Rejected / Not recommended",
    ]

    bucket_vals = {}
    for b in default_bucket_order:
        if b.startswith("1 "): default = 1.00
        elif b.startswith("2 "): default = 0.75
        elif b.startswith("3 "): default = 0.50
        elif b.startswith("4 "): default = 0.25
        else: default = 0.00

        bucket_vals[b] = st.slider(
            f"Bucket weight: {b}",
            min_value=0.0,
            max_value=1.0,
            value=default,
            step=0.05
        )

    st.markdown("---")
    st.markdown("### Funding band weights (0 → 1)")

    band_vals = {}
    if "Funding_Band" in dec_df.columns:
        for fb in sorted(dec_df["Funding_Band"].dropna().unique()):
            band_vals[fb] = st.slider(
                f"Funding band weight: {fb}",
                0.0, 1.0, 0.5, 0.05
            )
    else:
        st.info("No funding band info found.")
    
    # ---------- Normalize Final_Total ----------
    if "Final_Total" in dec_df.columns:
        s_min = dec_df["Final_Total"].min()
        s_max = dec_df["Final_Total"].max()
        dec_df["Score_norm"] = (dec_df["Final_Total"] - s_min) / (s_max - s_min)
    else:
        dec_df["Score_norm"] = 0

    # ---------- Assign bucket/band numeric weights ----------
    dec_df["Bucket_norm"] = dec_df["Bucket"].map(bucket_vals).fillna(0)
    dec_df["Band_norm"]   = dec_df["Funding_Band"].map(band_vals).fillna(0)

    # ---------- Compute final decision score ----------
    dec_df["Decision_Score"] = (
        dec_df["Score_norm"]  * w_score +
        dec_df["Bucket_norm"] * w_bucket +
        dec_df["Band_norm"]   * w_band
    ) * 100   # scale to 0–100

    st.markdown("### Final Ranking (0–100)")
    top_n = st.slider("How many projects to display?", 5, min(50, len(dec_df)), 15)

    show_cols = [
        "Project_ID",
        "Project_Name",
        "Decision_Score",
        "Final_Total",
        "Bucket",
        "Funding_Band",
        "Budget_EUR",
    ]

    st.dataframe(
        dec_df.sort_values("Decision_Score", ascending=False).head(top_n)[show_cols],
        use_container_width=True
    )

    st.caption(
        "Decision Score = normalized Final Score × weight_score "
        "+ Bucket_weight × weight_bucket "
        "+ Band_weight × weight_band"
    )


# ---------- TAB 8: BUCKETS & PRIORITIZATION ----------
# ---------- TAB 8: BUCKETS & PRIORITIZATION ----------
# ---------- TAB 8: BUCKETS & PRIORITIZATION ----------
with tab_buckets:
    st.subheader("🏷️ Buckets & prioritization")

    bucket_options = [
        "1 - Priority multi-sport Paralympic",
        "2 - Priority one-sport Paralympic",
        "3 - Other para sports",
        "4 - Others",
        "5 - Rejected / Not recommended",
    ]

    state_options = ["", "Approved", "Revision", "3rd review needed"]

    st.markdown("You can edit buckets here, and optionally import/export assignments as CSV.")

    # ---- CSV upload to resume work ----
    upload_bucket_file = st.file_uploader(
        "Upload previous bucket assignments (CSV with at least Project_ID and Bucket)",
        type="csv",
        key="bucket_assignments_upload",
    )

    # Base dataframe for current filtered projects
    base_cols = ["Project_ID", "Project_Name", "Budget_EUR", "Final_Total", "Multi_Sport", "Category", "Group"]
    base_cols = [c for c in base_cols if c in filtered_df.columns]
    base_df = filtered_df[base_cols].copy()

    # ---- Build or align bucket_df in session_state ----
    if "bucket_df" in st.session_state:
        prev = st.session_state["bucket_df"]
        if "Project_ID" in prev.columns:
            prev = prev[["Project_ID", "Bucket", "Flag", "State"]].copy()
            prev["Project_ID"] = prev["Project_ID"].astype(str)
            base_df["Project_ID"] = base_df["Project_ID"].astype(str)

            merged = base_df.merge(
                prev,
                on="Project_ID",
                how="left",
                suffixes=("", "_prev"),
            )

            # If no previous bucket, infer; otherwise keep old
            merged["Bucket"] = merged["Bucket"].where(
                merged["Bucket"].notna(),
                merged.apply(infer_bucket, axis=1),
            )

            # Flags / states default
            merged["Flag"] = merged["Flag"].fillna(False)
            merged["State"] = merged["State"].fillna("")

            bucket_df = merged[base_cols + ["Bucket", "Flag", "State"]].copy()
        else:
            # Fallback if prev somehow corrupted
            bucket_df = base_df.copy()
            bucket_df["Bucket"] = bucket_df.apply(infer_bucket, axis=1)
            bucket_df["Flag"] = False
            bucket_df["State"] = ""
    else:
        # First time: infer buckets
        bucket_df = base_df.copy()
        bucket_df["Bucket"] = bucket_df.apply(infer_bucket, axis=1)
        bucket_df["Flag"] = False
        bucket_df["State"] = ""

    # ---- If CSV uploaded, override Bucket/Flag/State where IDs match ----
    # ---- If CSV uploaded, override Bucket/Flag/State where IDs match ----
    if upload_bucket_file is not None:
        prev_csv = pd.read_csv(upload_bucket_file)

        # We need at least Project_ID and Bucket
        if "Project_ID" in prev_csv.columns and "Bucket" in prev_csv.columns:
            prev_csv["Project_ID"] = prev_csv["Project_ID"].astype(str)
            bucket_df["Project_ID"] = bucket_df["Project_ID"].astype(str)

            # Build list of columns that actually exist in the CSV
            merge_cols = ["Project_ID", "Bucket"]
            if "Flag" in prev_csv.columns:
                merge_cols.append("Flag")
            if "State" in prev_csv.columns:
                merge_cols.append("State")

            prev_use = prev_csv[merge_cols].copy()

            # Ensure Flag / State exist (even if they weren't in the CSV)
            if "Flag" not in prev_use.columns:
                prev_use["Flag"] = False
            if "State" not in prev_use.columns:
                prev_use["State"] = ""

            # Merge CSV info onto current bucket_df
            bucket_df = bucket_df.merge(
                prev_use,
                on="Project_ID",
                how="left",
                suffixes=("", "_csv"),
            )

            # Override bucket/flag/state with CSV values where they exist
            for col in ["Bucket", "Flag", "State"]:
                csv_col = col + "_csv"
                if csv_col in bucket_df.columns:
                    bucket_df[col] = bucket_df[csv_col].combine_first(bucket_df[col])

            # Clean up temporary _csv columns
            bucket_df.drop(
                columns=[c for c in bucket_df.columns if c.endswith("_csv")],
                inplace=True,
            )

            st.success("Loaded previous bucket assignments from CSV (for matching Project_IDs).")
        else:
            st.warning("CSV must contain at least 'Project_ID' and 'Bucket' columns.")

            st.warning("CSV must contain at least 'Project_ID' and 'Bucket' columns.")

    # ---------- 1) Interactive editor ----------
    edited = st.data_editor(
        bucket_df,
        key="bucket_editor",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Bucket": st.column_config.SelectboxColumn(
                "Bucket",
                options=bucket_options,
                required=True,
            ),
            "Flag": st.column_config.CheckboxColumn(
                "Flag (selected?)",
                default=False,
            ),
            "State": st.column_config.SelectboxColumn(
                "State",
                options=state_options,
                required=False,
            ),
            "Budget_EUR": st.column_config.NumberColumn("Budget (EUR)", format="€%d"),
            "Final_Total": st.column_config.NumberColumn("Final Total", format="%.1f"),
        }
    )

    # 🔑 From here on, edited is the source of truth
    bucket_df = edited.copy()
    st.session_state["bucket_df"] = bucket_df


    # ---------- 2) Download current assignments ----------
    st.markdown("### Export current bucket assignments")
    export_cols = [c for c in ["Project_ID", "Project_Name", "Bucket", "Flag", "State", "Budget_EUR", "Final_Total"] if c in bucket_df.columns]
    export_csv = bucket_df[export_cols].to_csv(index=False)
    st.download_button(
        "Download bucket assignments CSV",
        data=export_csv,
        file_name="bucket_assignments.csv",
        mime="text/csv",
    )

    # ---------- 3) Summary per bucket (with states + flags) ----------
    if "Bucket" in bucket_df.columns:
        summary = (
            bucket_df
            .groupby("Bucket", dropna=False)
            .agg(
                n_projects=("Project_ID", "nunique"),
                total_budget=("Budget_EUR", "sum"),
                avg_score=("Final_Total", "mean"),
                n_approved=("State", lambda s: (s == "Approved").sum()),
                n_revision=("State", lambda s: (s == "Revision").sum()),
                n_third=("State", lambda s: (s == "3rd review needed").sum()),
                n_flagged=("Flag", lambda x: (x == True).sum()),
            )
            .reset_index()
        )

        st.markdown("### Bucket summary (totals, states, flagged)")

        summary_display = summary.copy()
        summary_display["total_budget"] = summary_display["total_budget"].fillna(0)
        summary_display["avg_score"] = summary_display["avg_score"].round(1)
        summary_display["total_budget_eur"] = summary_display["total_budget"].apply(
            lambda x: f"€{x:,.0f}"
        )

        summary_display = summary_display[
            [
                "Bucket",
                "n_projects",
                "total_budget_eur",
                "avg_score",
                "n_approved",
                "n_revision",
                "n_third",
                "n_flagged",
            ]
        ].rename(
            columns={
                "n_projects": "Total projects",
                "n_approved": "Approved",
                "n_revision": "Revision",
                "n_third": "3rd review needed",
                "n_flagged": "Flagged",
            }
        )

        st.dataframe(summary_display, use_container_width=True)

        # ---------- 4) Bucket board (projects inside each bucket) ----------
        st.markdown("### Bucket board (projects inside each bucket)")

        cols_vis = st.columns(5)
        bucket_order = bucket_options  # fixed order 1–5

        # Map states to emoji labels
        state_icon_map = {
            "Approved": "✅ Approved",
            "Revision": "🟠 Revision",
            "3rd review needed": "🔍 3rd review needed",
            "": "",
            None: "",
        }

        for bucket_label, col in zip(bucket_order, cols_vis):
            with col:
                col.markdown(f"**{bucket_label}**")

                subset = bucket_df[bucket_df["Bucket"] == bucket_label].copy()
                if subset.empty:
                    col.caption("No projects assigned.")
                else:
                    # Format budget & score
                    subset["Budget_EUR_fmt"] = subset["Budget_EUR"].apply(
                        lambda x: f"€{x:,.0f}" if pd.notna(x) else "—"
                    )
                    subset["Final_Total_fmt"] = subset["Final_Total"].apply(
                        lambda x: f"{x:.1f}" if pd.notna(x) else "—"
                    )

                    # Emoji state label
                    subset["State_display"] = subset["State"].map(state_icon_map)

                    cols_show = [
                        "Project_ID",
                        "Project_Name",
                        "Group",
                        "Flag",
                        "State_display",
                        "Budget_EUR_fmt",
                        "Final_Total_fmt",
                    ]
                    cols_show = [c for c in cols_show if c in subset.columns]

                    display_df = subset[cols_show].rename(
                        columns={
                            "Project_ID": "ID",
                            "Project_Name": "Project",
                            "Group": "Group",
                            "Flag": "Flag",
                            "State_display": "State",
                            "Budget_EUR_fmt": "Budget",
                            "Final_Total_fmt": "Total points",
                        }
                    )

                    col.dataframe(display_df, use_container_width=True)

                    # Per-bucket totals for this column
                    if "Budget_EUR" in subset.columns:
                        total_b = subset["Budget_EUR"].sum()
                        b_approved = subset.loc[subset["State"] == "Approved", "Budget_EUR"].sum()
                        b_revision = subset.loc[subset["State"] == "Revision", "Budget_EUR"].sum()
                        b_third = subset.loc[subset["State"] == "3rd review needed", "Budget_EUR"].sum()
                        b_flagged = subset.loc[subset["Flag"] == True, "Budget_EUR"].sum()
                    else:
                        total_b = b_approved = b_revision = b_third = b_flagged = 0

                    n_total = len(subset)
                    n_approved = (subset["State"] == "Approved").sum()
                    n_revision = (subset["State"] == "Revision").sum()
                    n_third = (subset["State"] == "3rd review needed").sum()
                    n_flagged = subset["Flag"].sum() if "Flag" in subset.columns else 0

                    # 1) Total budget & project counts
                    col.caption(
                        f"Total budget (all): €{total_b:,.0f} | Projects: {n_total}"
                    )

                    # 2) Budget per state
                    col.caption(
                        f"Budget by state → ✅ €{b_approved:,.0f} | 🟠 €{b_revision:,.0f} | 🔍 €{b_third:,.0f}"
                    )

                    # 3) Flagged projects & their budget
                    col.caption(
                        f"Flagged: {int(n_flagged)} projects | Flagged budget: €{b_flagged:,.0f}"
                    )
    else:
        st.info("No bucket information available yet.")

# ---------- NEW TAB: BUCKETS BY FUNDING BAND ----------
# ---------- NEW TAB: BUCKETS BY FUNDING BAND ----------
with tab_bucket_bands:
    st.subheader("💶 Buckets by funding band")

    st.markdown(
        "Upload a bucket-assignments CSV (or reuse the current bucket table) "
        "to see how each bucket is distributed across funding bands."
    )

    # 1) Option to load a CSV with bucket assignments
    upload_band_file = st.file_uploader(
        "Upload bucket assignments CSV (same format as the export from the Buckets tab)",
        type="csv",
        key="bucket_band_csv",
    )

    band_df = None

    if upload_band_file is not None:
        # Use the uploaded CSV
        band_df = pd.read_csv(upload_band_file)
        st.success("Loaded bucket assignments from uploaded CSV.")
    elif "bucket_df" in st.session_state:
        # Fallback: use current in-memory bucket_df
        band_df = st.session_state["bucket_df"].copy()
        st.info("Using current in-memory bucket assignments from the Buckets tab.")
    else:
        st.warning(
            "No bucket assignments available yet. Go to 'Buckets & prioritization' first, or upload a CSV."
        )
        st.stop()

    # Ensure Project_ID is string for joining
    if "Project_ID" in band_df.columns:
        band_df["Project_ID"] = band_df["Project_ID"].astype(str)
    if "Project_ID" in df.columns:
        df["Project_ID"] = df["Project_ID"].astype(str)

    # 2) Bring in Funding_Band and Budget_EUR from the main dataframe if missing
    merge_cols = ["Project_ID"]
    if "Funding_Band" in df.columns and "Funding_Band" not in band_df.columns:
        merge_cols.append("Funding_Band")
    if "Budget_EUR" in df.columns and "Budget_EUR" not in band_df.columns:
        merge_cols.append("Budget_EUR")
    if "Final_Total" in df.columns and "Final_Total" not in band_df.columns:
        merge_cols.append("Final_Total")

    if len(merge_cols) > 1:
        meta = df[merge_cols].drop_duplicates("Project_ID")
        band_df = band_df.merge(meta, on="Project_ID", how="left")

    # Basic sanity: keep only rows with a bucket
    if "Bucket" not in band_df.columns:
        st.error("The bucket data does not contain a 'Bucket' column.")
        st.stop()

    band_df = band_df[band_df["Bucket"].notna()]

    if band_df.empty:
        st.warning("No projects with bucket assignments found.")
        st.stop()

    # 3) Summary Bucket × Funding_Band
    if "Funding_Band" not in band_df.columns:
        st.warning("No 'Funding_Band' column available in the data to split by.")
        st.stop()

    st.markdown("### Bucket × Funding band summary")

    summary = (
        band_df
        .groupby(["Bucket", "Funding_Band"], dropna=False)
        .agg(
            n_projects=("Project_ID", "nunique"),
            total_budget=("Budget_EUR", "sum"),
            avg_score=("Final_Total", "mean"),
        )
        .reset_index()
    )

    summary["total_budget"] = summary["total_budget"].fillna(0)
    summary["avg_score"] = summary["avg_score"].round(1)
    summary["total_budget_eur"] = summary["total_budget"].apply(lambda x: f"€{x:,.0f}")

    summary_display = summary[
        ["Bucket", "Funding_Band", "n_projects", "total_budget_eur", "avg_score"]
    ].rename(
        columns={
            "n_projects": "Projects",
            "total_budget_eur": "Total budget",
            "avg_score": "Avg score",
        }
    )

    st.dataframe(summary_display, use_container_width=True)

    # 4) Per-bucket, per-band project-level breakdown
    st.markdown("### Per-bucket funding band breakdown")

    for bucket_label in sorted(summary["Bucket"].dropna().unique()):
        st.markdown(f"#### {bucket_label}")

        # ---- Aggregated view for this bucket (Bucket × Funding_Band) ----
        sub_summary = summary_display[summary_display["Bucket"] == bucket_label].copy()
        sub_summary = sub_summary.drop(columns=["Bucket"])
        st.dataframe(sub_summary, use_container_width=True)

        # ---- Project-level view for this bucket, split by funding band ----
        bucket_projects = band_df[band_df["Bucket"] == bucket_label].copy()
        if bucket_projects.empty:
            st.caption("No projects in this bucket.")
            continue

        # Ensure key columns exist
        cols_base = [
            "Project_ID",
            "Project_Name",
            "Funding_Band",
            "Budget_EUR",
            "Final_Total",
            "Flag",
            "State",
        ]
        cols_base = [c for c in cols_base if c in bucket_projects.columns]
        bucket_projects = bucket_projects[cols_base].copy()

        # Format budget & score for nicer display
        if "Budget_EUR" in bucket_projects.columns:
            bucket_projects["Budget"] = bucket_projects["Budget_EUR"].apply(
                lambda x: f"€{x:,.0f}" if pd.notna(x) else "—"
            )
        else:
            bucket_projects["Budget"] = "—"

        if "Final_Total" in bucket_projects.columns:
            bucket_projects["Total_points"] = bucket_projects["Final_Total"].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else "—"
            )
        else:
            bucket_projects["Total_points"] = "—"

        # Order by Funding_Band then score (if available)
        sort_cols = []
        if "Funding_Band" in bucket_projects.columns:
            sort_cols.append("Funding_Band")
        if "Final_Total" in bucket_projects.columns:
            sort_cols.append("Final_Total")

        if sort_cols:
            bucket_projects = bucket_projects.sort_values(
                sort_cols, ascending=[True] * len(sort_cols)
            )

        # Build final display table
        display_cols = []
        for c in [
            "Project_ID",
            "Project_Name",
            "Funding_Band",
            "Flag",
            "State",
            "Budget",
            "Total_points",
        ]:
            if c in bucket_projects.columns:
                display_cols.append(c)

        proj_display = bucket_projects[display_cols].rename(
            columns={
                "Project_ID": "ID",
                "Project_Name": "Project title",
                "Funding_Band": "Band",
                "Total_points": "Total points",
            }
        )

        st.markdown("**Project-level details in this bucket (by funding band)**")
        st.dataframe(proj_display, use_container_width=True)

# ---------- NEW TAB: 4-BUCKET VIEW WITH EXTENDED STATES ----------
with tab_buckets4:
    st.subheader("🏷️ Buckets (4 only) & extended status")

    # 4 buckets only
    bucket_options = [
        "1 - Priority multi-sport Paralympic",
        "2 - Priority one-sport Paralympic",
        "3 - Other para sports",
        "4 - Others",
    ]

    # Extended state options
    state_options = ["", "Approved", "Pending", "Revision", "3rd review needed", "Rejected"]

    st.markdown(
        "This tab uses 4 buckets (no 'Rejected' bucket) and a richer status field:\n\n"
        "- **Bucket** = strategic category (1–4)\n"
        "- **State** = workflow status: Approved, Pending, Revision, 3rd review needed, Rejected\n\n"
        "You can edit buckets here, upload/download CSV, and see summaries."
    )

    # ---- CSV upload to resume work ----
    upload_bucket_file = st.file_uploader(
        "Upload previous 4-bucket assignments (CSV with at least Project_ID and Bucket)",
        type="csv",
        key="bucket4_assignments_upload",
    )

    # Base dataframe for current filtered projects
    base_cols = ["Project_ID", "Project_Name", "Budget_EUR", "Final_Total", "Multi_Sport", "Category", "Group"]
    base_cols = [c for c in base_cols if c in filtered_df.columns]
    base_df = filtered_df[base_cols].copy()

    # ---- Build or align bucket4_df in session_state ----
    if "bucket4_df" in st.session_state:
        prev = st.session_state["bucket4_df"]
        if "Project_ID" in prev.columns:
            prev = prev[["Project_ID", "Bucket", "Flag", "State"]].copy()
            prev["Project_ID"] = prev["Project_ID"].astype(str)
            base_df["Project_ID"] = base_df["Project_ID"].astype(str)

            merged = base_df.merge(
                prev,
                on="Project_ID",
                how="left",
                suffixes=("", "_prev"),
            )

            # If no previous bucket, infer; otherwise keep old
            merged["Bucket"] = merged["Bucket"].where(
                merged["Bucket"].notna(),
                merged.apply(infer_bucket, axis=1),
            )

            # Flags / states default
            merged["Flag"] = merged["Flag"].fillna(False)
            merged["State"] = merged["State"].fillna("")

            bucket_df = merged[base_cols + ["Bucket", "Flag", "State"]].copy()
        else:
            # Fallback if prev somehow corrupted
            bucket_df = base_df.copy()
            bucket_df["Bucket"] = bucket_df.apply(infer_bucket, axis=1)
            bucket_df["Flag"] = False
            bucket_df["State"] = ""
    else:
        # First time: infer buckets
        bucket_df = base_df.copy()
        bucket_df["Bucket"] = bucket_df.apply(infer_bucket, axis=1)
        bucket_df["Flag"] = False
        bucket_df["State"] = ""

    # Coerce any buckets not in the 4-bucket set into "4 - Others"
    valid_buckets = set(bucket_options)
    bucket_df["Bucket"] = bucket_df["Bucket"].where(
        bucket_df["Bucket"].isin(valid_buckets),
        "4 - Others",
    )

    # ---- If CSV uploaded, override Bucket/Flag/State where IDs match ----
    if upload_bucket_file is not None:
        prev_csv = pd.read_csv(upload_bucket_file)

        # We need at least Project_ID and Bucket
        if "Project_ID" in prev_csv.columns and "Bucket" in prev_csv.columns:
            prev_csv["Project_ID"] = prev_csv["Project_ID"].astype(str)
            bucket_df["Project_ID"] = bucket_df["Project_ID"].astype(str)

            # Build list of columns that actually exist in the CSV
            merge_cols = ["Project_ID", "Bucket"]
            if "Flag" in prev_csv.columns:
                merge_cols.append("Flag")
            if "State" in prev_csv.columns:
                merge_cols.append("State")

            prev_use = prev_csv[merge_cols].copy()

            # Ensure Flag / State exist (even if they weren't in the CSV)
            if "Flag" not in prev_use.columns:
                prev_use["Flag"] = False
            if "State" not in prev_use.columns:
                prev_use["State"] = ""

            # Merge CSV info onto current bucket_df
            bucket_df = bucket_df.merge(
                prev_use,
                on="Project_ID",
                how="left",
                suffixes=("", "_csv"),
            )

            # Override bucket/flag/state with CSV values where they exist
            for col in ["Bucket", "Flag", "State"]:
                csv_col = col + "_csv"
                if csv_col in bucket_df.columns:
                    bucket_df[col] = bucket_df[csv_col].combine_first(bucket_df[col])

            # Clean up temporary _csv columns
            bucket_df.drop(
                columns=[c for c in bucket_df.columns if c.endswith("_csv")],
                inplace=True,
            )

            # After loading, again coerce any bucket not in the 4-bucket set to "4 - Others"
            bucket_df["Bucket"] = bucket_df["Bucket"].where(
                bucket_df["Bucket"].isin(valid_buckets),
                "4 - Others",
            )

            st.success("Loaded previous 4-bucket assignments from CSV (for matching Project_IDs).")
        else:
            st.warning("CSV must contain at least 'Project_ID' and 'Bucket' columns.")

    # ---------- 1) Interactive editor ----------
    edited = st.data_editor(
        bucket_df,
        key="bucket4_editor",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Bucket": st.column_config.SelectboxColumn(
                "Bucket",
                options=bucket_options,
                required=True,
            ),
            "Flag": st.column_config.CheckboxColumn(
                "Flag (selected?)",
                default=False,
            ),
            "State": st.column_config.SelectboxColumn(
                "State",
                options=state_options,
                required=False,
            ),
            "Budget_EUR": st.column_config.NumberColumn("Budget (EUR)", format="€%d"),
            "Final_Total": st.column_config.NumberColumn("Final Total", format="%.1f"),
        }
    )

    # 🔑 From here on, edited is the source of truth for this 4-bucket view
    bucket_df = edited.copy()
    st.session_state["bucket4_df"] = bucket_df

    # ---------- 2) Download current assignments ----------
    st.markdown("### Export current 4-bucket assignments")
    export_cols = [
        c
        for c in ["Project_ID", "Project_Name", "Bucket", "Flag", "State", "Budget_EUR", "Final_Total"]
        if c in bucket_df.columns
    ]
    export_csv = bucket_df[export_cols].to_csv(index=False)
    st.download_button(
        "Download 4-bucket assignments CSV",
        data=export_csv,
        file_name="bucket4_assignments.csv",
        mime="text/csv",
    )

    # ---------- 3) Summary per bucket (with states + flags) ----------
    if "Bucket" in bucket_df.columns:
        summary = (
            bucket_df
            .groupby("Bucket", dropna=False)
            .agg(
                n_projects=("Project_ID", "nunique"),
                total_budget=("Budget_EUR", "sum"),
                avg_score=("Final_Total", "mean"),
                n_approved=("State", lambda s: (s == "Approved").sum()),
                n_pending=("State", lambda s: (s == "Pending").sum()),
                n_revision=("State", lambda s: (s == "Revision").sum()),
                n_third=("State", lambda s: (s == "3rd review needed").sum()),
                n_rejected=("State", lambda s: (s == "Rejected").sum()),
                n_flagged=("Flag", lambda x: (x == True).sum()),
            )
            .reset_index()
        )

        st.markdown("### 4-bucket summary (totals, states, flagged)")

        summary_display = summary.copy()
        summary_display["total_budget"] = summary_display["total_budget"].fillna(0)
        summary_display["avg_score"] = summary_display["avg_score"].round(1)
        summary_display["total_budget_eur"] = summary_display["total_budget"].apply(
            lambda x: f"€{x:,.0f}"
        )

        summary_display = summary_display[
            [
                "Bucket",
                "n_projects",
                "total_budget_eur",
                "avg_score",
                "n_approved",
                "n_pending",
                "n_revision",
                "n_third",
                "n_rejected",
                "n_flagged",
            ]
        ].rename(
            columns={
                "n_projects": "Total projects",
                "n_approved": "Approved",
                "n_pending": "Pending",
                "n_revision": "Revision",
                "n_third": "3rd review",
                "n_rejected": "Rejected",
                "n_flagged": "Flagged",
            }
        )

        st.dataframe(summary_display, use_container_width=True)

        # ---------- 4) Bucket board (projects inside each bucket) ----------
       # ---------- NEW TAB: 4-BUCKET VIEW WITH EXTENDED STATES ----------
# ---------- NEW TAB: 4-BUCKET VIEW WITH EXTENDED STATES ----------
with tab_buckets4:
    st.subheader("🏷️ Buckets (4 only) & extended status")

    # 4 buckets only
    bucket_options = [
        "1 - Priority multi-sport Paralympic",
        "2 - Priority one-sport Paralympic",
        "3 - Other para sports",
        "4 - Others",
    ]

    # Extended state options
    state_options = ["", "Approved", "Pending", "Revision", "3rd review needed", "Rejected"]

    st.markdown(
        "This tab uses 4 buckets (no 'Rejected' bucket) and a richer status field:\n\n"
        "- **Bucket** = strategic category (1–4)\n"
        "- **State** = workflow status: Approved, Pending, Revision, 3rd review needed, Rejected\n\n"
        "You can edit buckets here, upload/download CSV, and see summaries."
    )

    # ---- CSV upload to resume work ----
    upload_bucket_file = st.file_uploader(
        "Upload previous 4-bucket assignments (CSV with at least Project_ID and Bucket)",
        type="csv",
        key="bucket4_assignments_upload",
    )

    # Base dataframe for current filtered projects
    base_cols = [
        "Project_ID",
        "Project_Name",
        "Budget_EUR",
        "Final_Total",
        "Multi_Sport",
        "Category",
        "Group",
    ]
    base_cols = [c for c in base_cols if c in filtered_df.columns]
    base_df = filtered_df[base_cols].copy()

    # ---- Build or align bucket4_df in session_state ----
    if "bucket4_df" in st.session_state:
        prev = st.session_state["bucket4_df"]
        if "Project_ID" in prev.columns:
            prev = prev[["Project_ID", "Bucket", "Flag", "State"]].copy()
            prev["Project_ID"] = prev["Project_ID"].astype(str)
            base_df["Project_ID"] = base_df["Project_ID"].astype(str)

            merged = base_df.merge(
                prev,
                on="Project_ID",
                how="left",
                suffixes=("", "_prev"),
            )

            # If no previous bucket, infer; otherwise keep old
            merged["Bucket"] = merged["Bucket"].where(
                merged["Bucket"].notna(),
                merged.apply(infer_bucket, axis=1),
            )

            # Flags / states default
            merged["Flag"] = merged["Flag"].fillna(False)
            merged["State"] = merged["State"].fillna("")

            bucket_df = merged[base_cols + ["Bucket", "Flag", "State"]].copy()
        else:
            # Fallback if prev somehow corrupted
            bucket_df = base_df.copy()
            bucket_df["Bucket"] = bucket_df.apply(infer_bucket, axis=1)
            bucket_df["Flag"] = False
            bucket_df["State"] = ""
    else:
        # First time: infer buckets
        bucket_df = base_df.copy()
        bucket_df["Bucket"] = bucket_df.apply(infer_bucket, axis=1)
        bucket_df["Flag"] = False
        bucket_df["State"] = ""

    # Coerce any buckets not in the 4-bucket set into "4 - Others"
    valid_buckets = set(bucket_options)
    bucket_df["Bucket"] = bucket_df["Bucket"].where(
        bucket_df["Bucket"].isin(valid_buckets),
        "4 - Others",
    )

    # ---- If CSV uploaded, override Bucket/Flag/State where IDs match ----
    if upload_bucket_file is not None:
        prev_csv = pd.read_csv(upload_bucket_file)

        # We need at least Project_ID and Bucket
        if "Project_ID" in prev_csv.columns and "Bucket" in prev_csv.columns:
            prev_csv["Project_ID"] = prev_csv["Project_ID"].astype(str)
            bucket_df["Project_ID"] = bucket_df["Project_ID"].astype(str)

            # Build list of columns that actually exist in the CSV
            merge_cols = ["Project_ID", "Bucket"]
            if "Flag" in prev_csv.columns:
                merge_cols.append("Flag")
            if "State" in prev_csv.columns:
                merge_cols.append("State")

            prev_use = prev_csv[merge_cols].copy()

            # Ensure Flag / State exist (even if they weren't in the CSV)
            if "Flag" not in prev_use.columns:
                prev_use["Flag"] = False
            if "State" not in prev_use.columns:
                prev_use["State"] = ""

            # Merge CSV info onto current bucket_df
            bucket_df = bucket_df.merge(
                prev_use,
                on="Project_ID",
                how="left",
                suffixes=("", "_csv"),
            )

            # Override bucket/flag/state with CSV values where they exist
            for col in ["Bucket", "Flag", "State"]:
                csv_col = col + "_csv"
                if csv_col in bucket_df.columns:
                    bucket_df[col] = bucket_df[csv_col].combine_first(bucket_df[col])

            # Clean up temporary _csv columns
            bucket_df.drop(
                columns=[c for c in bucket_df.columns if c.endswith("_csv")],
                inplace=True,
            )

            # After loading, again coerce buckets not in the 4-bucket set to "4 - Others"
            bucket_df["Bucket"] = bucket_df["Bucket"].where(
                bucket_df["Bucket"].isin(valid_buckets),
                "4 - Others",
            )

            st.success("Loaded previous 4-bucket assignments from CSV (for matching Project_IDs).")
        else:
            st.warning("CSV must contain at least 'Project_ID' and 'Bucket' columns.")

    # ---------- 1) Interactive editor ----------
    edited = st.data_editor(
        bucket_df,
        key="bucket4_editor",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Bucket": st.column_config.SelectboxColumn(
                "Bucket",
                options=bucket_options,
                required=True,
            ),
            "Flag": st.column_config.CheckboxColumn(
                "Flag (selected?)",
                default=False,
            ),
            "State": st.column_config.SelectboxColumn(
                "State",
                options=state_options,
                required=False,
            ),
            "Budget_EUR": st.column_config.NumberColumn(
                "Budget (EUR)",
                format="€%d",
            ),
            "Final_Total": st.column_config.NumberColumn(
                "Final Total",
                format="%.1f",
            ),
        }
    )

    # 🔑 From here on, edited is the source of truth for this 4-bucket view
    bucket_df = edited.copy()
    st.session_state["bucket4_df"] = bucket_df

    # 🔁 Sync 4-bucket assignments to the global bucket_df
    # so other tabs (decision support, bands, etc.) also see these updates
    st.session_state["bucket_df"] = bucket_df

    # ---------- 2) Download current assignments ----------
    st.markdown("### Export current 4-bucket assignments")
    export_cols = [
        c
        for c in ["Project_ID", "Project_Name", "Bucket", "Flag", "State", "Budget_EUR", "Final_Total"]
        if c in bucket_df.columns
    ]
    export_csv = bucket_df[export_cols].to_csv(index=False)
    st.download_button(
        "Download 4-bucket assignments CSV",
        data=export_csv,
        file_name="bucket4_assignments.csv",
        mime="text/csv",
    )

    # ---------- 3) Summary per bucket (with states + flags) ----------
    if "Bucket" in bucket_df.columns:
        summary = (
            bucket_df
            .groupby("Bucket", dropna=False)
            .agg(
                n_projects=("Project_ID", "nunique"),
                total_budget=("Budget_EUR", "sum"),
                avg_score=("Final_Total", "mean"),
                n_approved=("State", lambda s: (s == "Approved").sum()),
                n_pending=("State", lambda s: (s == "Pending").sum()),
                n_revision=("State", lambda s: (s == "Revision").sum()),
                n_third=("State", lambda s: (s == "3rd review needed").sum()),
                n_rejected=("State", lambda s: (s == "Rejected").sum()),
                n_flagged=("Flag", lambda x: (x == True).sum()),
            )
            .reset_index()
        )

        st.markdown("### 4-bucket summary (totals, states, flagged)")

        summary_display = summary.copy()
        summary_display["total_budget"] = summary_display["total_budget"].fillna(0)
        summary_display["avg_score"] = summary_display["avg_score"].round(1)
        summary_display["total_budget_eur"] = summary_display["total_budget"].apply(
            lambda x: f"€{x:,.0f}"
        )

        summary_display = summary_display[
            [
                "Bucket",
                "n_projects",
                "total_budget_eur",
                "avg_score",
                "n_approved",
                "n_pending",
                "n_revision",
                "n_third",
                "n_rejected",
                "n_flagged",
            ]
        ].rename(
            columns={
                "n_projects": "Total projects",
                "n_approved": "Approved",
                "n_pending": "Pending",
                "n_revision": "Revision",
                "n_third": "3rd review",
                "n_rejected": "Rejected",
                "n_flagged": "Flagged",
            }
        )

        st.dataframe(summary_display, use_container_width=True)

        # ---------- 4) Bucket board (projects inside each bucket) ----------
        st.markdown("### 4-bucket board (projects inside each bucket)")

        cols_vis = st.columns(4)
        bucket_order = bucket_options  # fixed order 1–4

        # Map states to emoji labels
        state_icon_map = {
            "Approved": "✅ Approved",
            "Pending": "⏳ Pending",
            "Revision": "🟠 Revision",
            "3rd review needed": "🔍 3rd review",
            "Rejected": "❌ Rejected",
            "": "",
            None: "",
        }

        for bucket_label, col in zip(bucket_order, cols_vis):
            with col:
                col.markdown(f"**{bucket_label}**")

                subset = bucket_df[bucket_df["Bucket"] == bucket_label].copy()
                if subset.empty:
                    col.caption("No projects assigned.")
                else:
                    # Format budget & score
                    subset["Budget_EUR_fmt"] = subset["Budget_EUR"].apply(
                        lambda x: f"€{x:,.0f}" if pd.notna(x) else "—"
                    )
                    subset["Final_Total_fmt"] = subset["Final_Total"].apply(
                        lambda x: f"{x:.1f}" if pd.notna(x) else "—"
                    )

                    # Emoji state label
                    subset["State_display"] = subset["State"].map(state_icon_map)

                    cols_show = [
                        "Project_ID",
                        "Project_Name",
                        "Group",
                        "Flag",
                        "State_display",
                        "Budget_EUR_fmt",
                        "Final_Total_fmt",
                    ]
                    cols_show = [c for c in cols_show if c in subset.columns]

                    display_df = subset[cols_show].rename(
                        columns={
                            "Project_ID": "ID",
                            "Project_Name": "Project",
                            "Group": "Group",
                            "Flag": "Flag",
                            "State_display": "State",
                            "Budget_EUR_fmt": "Budget",
                            "Final_Total_fmt": "Total points",
                        }
                    )

                    col.dataframe(display_df, use_container_width=True)

                    # Per-bucket totals
                    if "Budget_EUR" in subset.columns:
                        total_b = subset["Budget_EUR"].sum()
                        b_approved = subset.loc[subset["State"] == "Approved", "Budget_EUR"].sum()
                        b_pending = subset.loc[subset["State"] == "Pending", "Budget_EUR"].sum()
                        b_revision = subset.loc[subset["State"] == "Revision", "Budget_EUR"].sum()
                        b_third = subset.loc[subset["State"] == "3rd review needed", "Budget_EUR"].sum()
                        b_rejected = subset.loc[subset["State"] == "Rejected", "Budget_EUR"].sum()
                        b_flagged = subset.loc[subset["Flag"] == True, "Budget_EUR"].sum()
                    else:
                        total_b = b_approved = b_pending = b_revision = b_third = b_rejected = b_flagged = 0

                    n_total = len(subset)
                    n_approved = (subset["State"] == "Approved").sum()
                    n_pending = (subset["State"] == "Pending").sum()
                    n_revision = (subset["State"] == "Revision").sum()
                    n_third = (subset["State"] == "3rd review needed").sum()
                    n_rejected = (subset["State"] == "Rejected").sum()
                    n_flagged = subset["Flag"].sum() if "Flag" in subset.columns else 0

                    # 1) Total budget & project counts
                    col.caption(
                        f"Total budget (all): €{total_b:,.0f} | Projects: {n_total}"
                    )

                    # 2) Budget per state
                    col.caption(
                        "Budget by state → "
                        f"✅ €{b_approved:,.0f} | "
                        f"⏳ €{b_pending:,.0f} | "
                        f"🟠 €{b_revision:,.0f} | "
                        f"🔍 €{b_third:,.0f} | "
                        f"❌ €{b_rejected:,.0f}"
                    )

                    # 3) Flagged projects & their budget
                    col.caption(
                        f"Flagged: {int(n_flagged)} projects | Flagged budget: €{b_flagged:,.0f}"
                    )
    else:
        st.info("No bucket information available yet.")


# ---------- TAB 9: DRAG & DROP BUCKET BOARD ----------
with tab_dragdrop:
    from streamlit_sortables import sort_items

    st.subheader("🖱️ Drag & drop bucket assignment")

    bucket_options = [
        "1 - Priority multi-sport Paralympic",
        "2 - Priority one-sport Paralympic",
        "3 - Other para sports",
        "4 - Others",
        "5 - Rejected / Not recommended",
    ]

    # Base dataframe for visible projects
    base_cols = ["Project_ID", "Project_Name", "Budget_EUR", "Final_Total", "Multi_Sport", "Category", "Group"]
    base_cols = [c for c in base_cols if c in filtered_df.columns]
    base_df = filtered_df[base_cols].copy()
    base_df["Project_ID"] = base_df["Project_ID"].astype(str)

    # Start from bucket_df in session if present, else create fresh
    if "bucket_df" in st.session_state:
        prev = st.session_state["bucket_df"].copy()
        if "Project_ID" in prev.columns:
            prev["Project_ID"] = prev["Project_ID"].astype(str)
            bucket_df = base_df.merge(
                prev[["Project_ID", "Bucket", "Flag", "State"]],
                on="Project_ID",
                how="left",
                suffixes=("", "_prev"),
            )
            # If no bucket yet for this filtered subset, infer
            bucket_df["Bucket"] = bucket_df["Bucket"].where(
                bucket_df["Bucket"].notna(),
                bucket_df.apply(infer_bucket, axis=1),
            )
            bucket_df["Flag"] = bucket_df["Flag"].fillna(False)
            bucket_df["State"] = bucket_df["State"].fillna("")
        else:
            bucket_df = base_df.copy()
            bucket_df["Bucket"] = bucket_df.apply(infer_bucket, axis=1)
            bucket_df["Flag"] = False
            bucket_df["State"] = ""
    else:
        bucket_df = base_df.copy()
        bucket_df["Bucket"] = bucket_df.apply(infer_bucket, axis=1)
        bucket_df["Flag"] = False
        bucket_df["State"] = ""

    # Build label -> ID map
    label_to_id = {}
    def make_label(row):
        name = str(row["Project_Name"])
        budget = row["Budget_EUR"] if not pd.isna(row["Budget_EUR"]) else 0
        score = row["Final_Total"] if not pd.isna(row["Final_Total"]) else 0
        return f"{name} | €{budget:,.0f} | {score:.1f} pts"

    for _, row in bucket_df.iterrows():
        label = make_label(row)
        label_to_id[label] = row["Project_ID"]

    # Build containers for streamlit-sortables: list of dicts
    containers = []
    for b in bucket_options:
        subset = bucket_df[bucket_df["Bucket"] == b]
        labels = [make_label(row) for _, row in subset.iterrows()]
        containers.append(
            {
                "header": b,
                "items": labels,
            }
        )

    # Drag & drop UI
    sorted_containers = sort_items(
        containers,
        multi_containers=True,
        key="dragdrop_buckets",
    )

    # Update bucket_df from drag result
    for container in sorted_containers:
        bucket_label = container["header"]
        for label in container["items"]:
            pid = label_to_id.get(label)
            if pid is not None:
                bucket_df.loc[bucket_df["Project_ID"] == pid, "Bucket"] = bucket_label

    # Save back to session_state so Buckets tab sees the new assignments
    st.session_state["bucket_df"] = bucket_df

    st.success("Drag-and-drop updates saved. Check the 'Buckets & prioritization' tab for updated summaries.")
