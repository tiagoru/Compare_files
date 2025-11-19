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

tab_overview, tab_scores, tab_agreement, tab_risk, tab_profiles, tab_comments, tab_decision = st.tabs(
    [
        "📁 Overview",
        "📈 Scores & funding",
        "⚖️ Reviewer agreement",
        "🚥 Risk & sentiment",
        "🧬 Project profiles",
        "🗯️ Comment insights",
        "🧠 Decision support",
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
with tab_decision:
    st.subheader("🧠 Decision-support ranking")

    def col_or_zero(name: str) -> pd.Series:
        if name in filtered_df.columns:
            return filtered_df[name]
        return pd.Series(0, index=filtered_df.index)

    # Risk penalty: High = 10, Medium = 5, else 0
    if "Risk_Category" in filtered_df.columns:
        rc = filtered_df["Risk_Category"].fillna("").str.lower()
        risk_penalty = (
            (rc.str_contains("high").astype(int) * 10)
            + (rc.str_contains("medium").astype(int) * 5)
        )
    else:
        risk_penalty = pd.Series(0, index=filtered_df.index)

    decision_score = (
        col_or_zero("Final_Total") * 0.7 +
        col_or_zero("Innovation_avg") * 0.2 +
        col_or_zero("Impact_avg") * 0.1 -
        risk_penalty
    )

    filtered_df["Decision_Score"] = decision_score

    top_n = st.slider("Number of projects to display", 5, min(20, len(filtered_df)), 10)
    ranked = filtered_df.sort_values("Decision_Score", ascending=False).head(top_n)

    st.markdown("**Ranked projects (higher = more favourable)**")
    show_cols = [c for c in [
        "Project_ID", "Project_Name",
        "Decision_Score",
        "Final_Total",
        "Budget_EUR",
        "Methods_avg", "Impact_avg", "Innovation_avg", "Plan_avg", "Team_avg",
        "Risk_Category",
        "Funding_Band",
    ] if c in ranked.columns]
    st.dataframe(ranked[show_cols], use_container_width=True)

    st.caption(
        "Decision_Score = Final_Total (70%) + Innovation_avg (20%) + Impact_avg (10%) "
        "– penalty for Medium/High risk. Adjust the formula in the code if you want different weights."
    )
