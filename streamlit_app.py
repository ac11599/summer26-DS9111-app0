# --- Package Imports ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# dataset
# https://www.kaggle.com/datasets/desalegngeb/students-exam-scores

# --- Setup ---
st.set_page_config(
    page_title="Ad Revenue Dashboard 🌐",
    layout="centered",
    page_icon="🌐",)
st.sidebar.title("Ad Revenue 🌐")
page = st.sidebar.selectbox("Select Page", [
                            "Introduction 📘", "Visualization 📊", "Automated Report 📑", "Prediction 🔮"])

st.image("ad_revenue.jpg")

df = pd.read_csv("ads.csv")[['date', 'platform', 'campaign_type',
                             'industry', 'country', 'impressions', 'clicks', 'ad_spend', 'revenue']]

# --- Introduction Page ---
if page == "Introduction 📘":

    st.title("01 Introduction 📘")

    st.header("Data Preview")
    rows = st.slider("Select a number of rows to display", 5, 25, 5)
    st.dataframe(df.head(rows))

    st.header("Missing values")
    missing = df.isnull().sum()
    st.write(missing)

    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ you have missing values")

    st.header("📈 Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())

# --- Visualization Page ---
elif page == "Visualization 📊":

    st.title("02 Data Visualization 📊")

    # Axis options
    numeric_cols = ['impressions', 'clicks', 'ad_spend', 'revenue']
    categorical_cols = ['platform', 'campaign_type', 'industry', 'country']

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Trends Over Time 📈", "Performance by Category 📊", "Performance by Country 🌍", "Correlation Heatmap 🔥"])

    with tab1:
        st.header("Trends Over Time 📈")
        y_axis = st.selectbox("Select metric to plot over time",
                              numeric_cols, index=3, key="tab1_y")
        df['date'] = pd.to_datetime(df['date'])
        time_data = df.groupby('date')[y_axis].sum()
        st.line_chart(time_data, use_container_width=True)

    with tab2:
        st.header("Performance by Category 📊")
        col1, col2 = st.columns(2)
        with col1:
            cat_x = st.selectbox(
                "Select category", categorical_cols, index=0, key="tab2_x")
        with col2:
            cat_y = st.selectbox(
                "Select metric", numeric_cols, index=3, key="tab2_y")
        agg_func = st.selectbox(
                "Aggregate by", ["Sum", "Mean", "Median"], key="tab2_agg")
        agg_map = {"Sum": "sum", "Mean": "mean", "Median": "median"}
        cat_data = df.groupby(cat_x)[cat_y].agg(agg_map[agg_func])
        st.bar_chart(cat_data, use_container_width=True)

    with tab3:
        st.header("Performance by Country 🌍")
        country_metric = st.selectbox(
            "Select metric", numeric_cols, index=3, key="tab3_y")
        agg_func = st.selectbox(
                "Aggregate by", ["Sum", "Mean", "Median"], key="tab3_agg")
        agg_map = {"Sum": "sum", "Mean": "mean", "Median": "median"}
        country_data = df.groupby(
            'country')[country_metric].agg(agg_map[agg_func])
        st.bar_chart(country_data, use_container_width=True)

    with tab4:
        st.header("Correlation Heatmap 🔥")
        df_numeric = df[numeric_cols]
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_numeric.corr(), annot=True,
                    fmt=".2f", cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)

# --- Automated Report Page  (TO BE WORKED ON) ---
elif page == "Automated Report 📑":
    st.title("03 Automated Report")
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            profile = ProfileReport(
                df, title="Ads Revenue Report", explorative=True, minimal=True)
            st_profile_report(profile)

        export = profile.to_html()
        st.download_button(label="📥 Download full Report", data=export,
                           file_name="ads_revenue_report.html", mime='text/html')

# --- Prediction Page (TO BE WORKED ON) ---
elif page == "Prediction 🔮":
    st.title("04 Prediction (with Linear Regression) 🔮")

    # --- Data Preprocessing ---

    # Remove missing values

    df = df.dropna()

    # Define X and Y
    features_selection = st.sidebar.multiselect(
        "Select features (X)", ["impressions", "clicks", "ad_spend"], default=["impressions", "clicks", "ad_spend"])
    target = "revenue"

    X = df[features_selection]
    y = df[target]

    selected_metrics = st.sidebar.multiselect("Metrics to display", [
                                              "Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R² Score"], default=["Mean Absolute Error (MAE)"])

    # st.dataframe(X.head())
    # st.dataframe(y.head())

    # split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Model ---

    # Definition model
    model = LinearRegression()

    # Training model
    model.fit(X_train, y_train)

    # Prediction
    predictions = model.predict(X_test)

    # Evaluation
    if selected_metrics:
        st.header("Metrics")
        if "Mean Absolute Error (MAE)" in selected_metrics:
            mae = metrics.mean_absolute_error(y_test, predictions)
            st.write(f"- **MAE**: ${mae:,.2f}")
        if "Mean Squared Error (MSE)" in selected_metrics:
            mse = metrics.mean_squared_error(y_test, predictions)
            st.write(f"- **MSE**: ${mse:,.2f}")
        if "R² Score" in selected_metrics:
            r2 = metrics.r2_score(y_test, predictions)
            st.write(f"- **R2**: {r2:,.3f}")

    st.header("Actual vs. Predicted Revenue")
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()], "--r", linewidth=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)
