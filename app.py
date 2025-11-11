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
if "Research project title:" in df.columns:
    df["Project_Name"] = df["Research project title:"].astype(str)
elif "Project title2" in df.columns:
    df["Project_Name"] = df["Project title2"].astype(str)
else:
    df["Project_Name"] = df.iloc[:, 0].astype(str)

if "ID" in df.columns:
    df["Project_ID"] = df["ID"]

if "Please specify which of the funding bands is requested for the project?" in df.columns:
    df["Funding_Band"] = df["Please specify which of the funding bands is requested for the project?"]

if "Final Total" in df.columns:
    df["Final_Total"] = df["Final Total"]

# ===================== DERIVED SCORE COLUMNS =====================
score_pairs = [
    ("Methods", "Methods_46_review1", "Methods_46_review2"),
    ("Impact", "Impact_47_review1", "Impact_47_review2"),
    ("Innovation", "Innovation_48_review1", "Innovation_48_review2"),
    ("Plan", "Plan_49_review1", "Plan_49_review2"),
    ("Team", "Team_50_review1", "Team_50_review2"),
    ("Total", "Total_51_review1", "Total_51_review2"),
]
for label, c1, c2 in score_pairs:
    if c1 in df.columns and c2 in df.columns:
        df[f"{label}_avg"] = df[[c1, c2]].mean(axis=1)

# ===================== SIDEBAR FILTERS =====================
st.sidebar.header("🔍 Filters")

if "Risk_Category" in df.columns:
    risk_options = ["All"] + sorted(df["Risk_Category"].dropna().unique().tolist())
    risk_filter = st.sidebar.selectbox("Risk category", risk_options, index=0)
else:
    risk_filter = "All"

if "Funding_Band" in df.columns:
    band_options = ["All"] + sorted(df["Funding_Band"].dropna().unique().tolist())
    band_filter = st.sidebar.selectbox("Funding band", band_options, index=0)
else:
    band_filter = "All"

if "alert" in df.columns:
    alert_options = ["All"] + sorted(df["alert"].dropna().unique().tolist())
    alert_filter = st.sidebar.selectbox("Alert flag", alert_options, index=0)
else:
    alert_filter = "All"

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
with col3:
    if "Risk_Category" in filtered_df.columns:
        high_risk = filtered_df["Risk_Category"].fillna("").str.contains("High", case=False).sum()
        st.metric("High-risk projects", int(high_risk))
with col4:
    if "Funding_Band" in filtered_df.columns:
        st.metric("Funding bands used", int(filtered_df["Funding_Band"].nunique()))
st.markdown("---")

# ===================== TABS =====================
tab_overview, tab_scores, tab_agreement, tab_risk, tab_comments = st.tabs(
    ["📁 Overview", "📈 Score distribution", "⚖️ Reviewer agreement", "🚥 Risk & sentiment", "🗯️ Comment analysis"]
)

# ---------- TAB 1: OVERVIEW ----------
with tab_overview:
    st.subheader("Project overview")
    cols_to_show = [c for c in [
        "Project_ID","Project_Name","Final_Total","Methods_avg","Impact_avg",
        "Innovation_avg","Plan_avg","Team_avg","Risk_Category","Funding_Band","Project duration","alert"
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
                x="Project_Name", y=selected_score,
                color="Risk_Category" if "Risk_Category" in filtered_df.columns else None,
                hover_data=["Funding_Band"] if "Funding_Band" in filtered_df.columns else None,
                title=f"{selected_score} by project",
            )
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)

# ---------- TAB 3: REVIEWER AGREEMENT ----------
with tab_agreement:
    st.subheader("Reviewer agreement")
    dims_dict = {
        label: (c1, c2) for (label, c1, c2) in score_pairs if c1 in df.columns and c2 in df.columns
    }
    if not dims_dict:
        st.info("No reviewer 1 & 2 score pairs found.")
    else:
        dim = st.selectbox("Dimension", list(dims_dict.keys()))
        c1, c2 = dims_dict[dim]
        temp = filtered_df[["Project_Name", c1, c2]].dropna()
        temp["abs_diff"] = (temp[c1] - temp[c2]).abs()
        if not temp.empty:
            fig_diff = px.bar(
                temp.sort_values("abs_diff", ascending=False),
                x="Project_Name", y="abs_diff", hover_data=[c1, c2],
                title=f"Reviewer disagreement on {dim}",
            )
            fig_diff.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_diff, use_container_width=True)

# ---------- TAB 4: RISK & SENTIMENT ----------
with tab_risk:
    st.subheader("Risk and sentiment")
    if {"Final_Total", "Sentiment_Polarity"}.issubset(filtered_df.columns):
        fig_scatter = px.scatter(
            filtered_df,
            x="Final_Total", y="Sentiment_Polarity",
            color="Risk_Category" if "Risk_Category" in filtered_df.columns else None,
            hover_data=["Project_Name", "Funding_Band"] if "Funding_Band" in filtered_df.columns else ["Project_Name"],
            title="Final score vs sentiment polarity",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ---------- TAB 5: COMMENT ANALYSIS ----------
with tab_comments:
    st.subheader("🗯️ Word Cloud from comments")

    # Collect all comment-like columns
    comment_columns = [col for col in df.columns if "comment" in col.lower() or "feedback" in col.lower() or "Combined_Comments" in col]
    if not comment_columns:
        st.info("No comment columns found in the file.")
    else:
        st.write("The following columns were used for the text cloud:")
        st.write(", ".join(comment_columns))

        all_comments = ""
        for col in comment_columns:
            all_comments += " ".join(df[col].dropna().astype(str)) + " "

        if all_comments.strip() == "":
            st.warning("No comment text found to generate a word cloud.")
        else:
            wordcloud = WordCloud(
                width=1200, height=600, background_color="white", collocations=False
            ).generate(all_comments)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
