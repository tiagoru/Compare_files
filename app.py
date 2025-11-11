import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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

# ===================== COLUMN ALIASES / CLEANUP =====================

# Project title
if "Research project title:" in df.columns:
    df["Project_Name"] = df["Research project title:"].astype(str)
elif "Project title2" in df.columns:
    df["Project_Name"] = df["Project title2"].astype(str)
else:
    df["Project_Name"] = df.iloc[:, 0].astype(str)

# Project ID
if "ID" in df.columns:
    df["Project_ID"] = df["ID"]

# Funding band
if "Please specify which of the funding bands is requested for the project?" in df.columns:
    df["Funding_Band"] = df["Please specify which of the funding bands is requested for the project?"]

# Final total
if "Final Total" in df.columns:
    df["Final_Total"] = df["Final Total"]

# Risk + explanation (new names vs old)
if "Risk_Category" not in df.columns and "Risk" in df.columns:
    df["Risk_Category"] = df["Risk"]
if "Risk_Explanation" not in df.columns and "Explanation" in df.columns:
    df["Risk_Explanation"] = df["Explanation"]

# Sentiment columns
sentiment_num_col = None
if "Sentiment_Polarity" in df.columns:
    sentiment_num_col = "Sentiment_Polarity"
elif "Sentiment_Score" in df.columns:
    sentiment_num_col = "Sentiment_Score"

sentiment_label_col = "Sentiment_Label" if "Sentiment_Label" in df.columns else None

# ===================== DERIVED SCORE COLUMNS (AVERAGE OF REVIEWERS) =====================

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

# Minimum final score
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

# ===================== BIG DIFFERENCE FLAG (≥ 4 IN ANY OF 5 ITEMS) =====================

big_diff_cols = [c for c in [
    "diff_Methods", "diff_Impact", "diff_Innovation", "diff_Plan", "diff_Team"
] if c in filtered_df.columns]

if big_diff_cols:
    filtered_df["Any_big_diff"] = (filtered_df[big_diff_cols].abs() >= 4).any(axis=1)
else:
    filtered_df["Any_big_diff"] = False

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
        high_risk = filtered_df["Risk_Category"].fillna("").str.contains("High", case=False).sum()
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

tab_overview, tab_scores, tab_agreement, tab_risk, tab_comments = st.tabs(
    ["📁 Overview", "📈 Score distribution", "⚖️ Reviewer agreement", "🚥 Risk & sentiment", "🗯️ Comment analysis"]
)

# ---------- TAB 1: OVERVIEW ----------
with tab_overview:
    st.subheader("Project overview (after filters)")

    cols_to_show = [c for c in [
        "Project_ID", "Project_Name", "Final_Total",
        "Methods_avg", "Impact_avg", "Innovation_avg", "Plan_avg", "Team_avg",
        "Risk_Category", "Funding_Band", "Project duration", "alert",
    ] if c in filtered_df.columns]

    if "Final_Total" in cols_to_show:
        view_df = filtered_df[cols_to_show].sort_values("Final_Total", ascending=False)
    else:
        view_df = filtered_df[cols_to_show]

    st.dataframe(view_df, use_container_width=True)

# ---------- TAB 2: SCORE DISTRIBUTION ----------
with tab_scores:
    st.subheader("Score distribution")

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
    st.subheader("Reviewer agreement (projects with ≥ 4-point difference in any item)")

    # Only projects that have at least one big diff in Methods/Impact/Innovation/Plan/Team
    if "Any_big_diff" in filtered_df.columns:
        big_df = filtered_df[filtered_df["Any_big_diff"]]
    else:
        big_df = pd.DataFrame()

    if big_df.empty:
        st.info("No projects with ≥ 4-point difference in any of Methods / Impact / Innovation / Plan / Team.")
    else:
        # Dimension -> (reviewer1_col, reviewer2_col, diff_col)
        dim_info = {
            "Methods":    ("Methods_46_review1",    "Methods_46_review2",    "diff_Methods"),
            "Impact":     ("Impact_47_review1",     "Impact_47_review2",     "diff_Impact"),
            "Innovation": ("Innovation_48_review1", "Innovation_48_review2", "diff_Innovation"),
            "Plan":       ("Plan_49_review1",       "Plan_49_review2",       "diff_Plan"),
            "Team":       ("Team_50_review1",       "Team_50_review2",       "diff_Team"),
        }

        # Keep only dimensions that actually exist
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

            temp = big_df[["Project_ID", "Project_Name", c1, c2, cdiff]].dropna()
            temp["abs_diff"] = temp[cdiff].abs()
            temp = temp[temp["abs_diff"] >= 4]

            if temp.empty:
                st.info(f"No projects with ≥ 4-point difference in {dim}.")
            else:
                fig_diff = px.bar(
                    temp.sort_values("abs_diff", ascending=False),
                    x="Project_Name",
                    y="abs_diff",
                    hover_data=["Project_ID", c1, c2],
                    title=f"Reviewer disagreement on {dim} (diff ≥ 4)",
                )
                fig_diff.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_diff, use_container_width=True)

                st.markdown("**Projects with ≥ 4-point difference in this dimension:**")
                st.dataframe(
                    temp[["Project_ID", "Project_Name", c1, c2, "abs_diff"]]
                    .sort_values("abs_diff", ascending=False),
                    use_container_width=True,
                )

# ---------- TAB 4: RISK & SENTIMENT ----------
with tab_risk:
    st.subheader("Risk & sentiment overview")

    cols = [c for c in [
        "Project_ID", "Project_Name",
        "Risk_Category",
        sentiment_label_col,
        sentiment_num_col,
        "Final_Total",
    ] if c is not None and c in filtered_df.columns]

    if cols:
        st.markdown("**Projects with risk and sentiment:**")
        st.dataframe(filtered_df[cols], use_container_width=True)
    else:
        st.info("No risk/sentiment columns available.")

    # Scatter, if we have numeric sentiment and final score
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
        if sentiment_label_col:
            st.markdown(f"**Sentiment label:** {row.get(sentiment_label_col, 'N/A')}")
        if sentiment_num_col:
            st.markdown(f"**Sentiment score:** {row.get(sentiment_num_col, 'N/A')}")
        st.markdown(f"**Final total:** {row.get('Final_Total', 'N/A')}")

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

# ---------- TAB 5: COMMENT ANALYSIS ----------
with tab_comments:
    st.subheader("🗯️ Comment analysis (word cloud)")

    # Identify comment/feedback/text columns
    comment_columns = []
    for col in df.columns:
        lc = col.lower()
        if any(k in lc for k in ["comment", "feedback", "text"]):
            comment_columns.append(col)

    # Ensure these key ones are included if present
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
            project_options += (
                filtered_df[["Project_Name", "Project_ID"]]
                .dropna()
                .drop_duplicates()
                .apply(lambda r: f"{r['Project_Name']} (ID: {r['Project_ID']})", axis=1)
                .tolist()
            )

        proj_choice = st.selectbox("Select project", project_options)

        subset = filtered_df
        if (
            proj_choice != "All projects"
            and "Project_Name" in filtered_df.columns
            and "Project_ID" in filtered_df.columns
        ):
            # Parse the ID from "Title (ID: xxx)"
            try:
                proj_id_str = proj_choice.split("ID:")[1].rstrip(")")
                proj_id_str = proj_id_str.strip()
                subset = filtered_df[filtered_df["Project_ID"].astype(str) == proj_id_str]
            except Exception:
                subset = filtered_df[filtered_df["Project_Name"] == proj_choice]

        all_text = " ".join(subset[col_choice].dropna().astype(str))

        if not all_text.strip():
            st.warning("No text in this column for the selected project(s).")
        else:
            wordcloud = WordCloud(
                width=1200, height=600, background_color="white", collocations=False
            ).generate(all_text)

            st.markdown("**Word cloud**")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            if proj_choice == "All projects":
                st.caption("Text aggregated across all filtered projects.")
            else:
                st.caption(f"Text for: {proj_choice}")
