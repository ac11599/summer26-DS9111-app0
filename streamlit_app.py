# --- Package Imports ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# from ydata_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report

# dataset
# https://www.kaggle.com/datasets/desalegngeb/students-exam-scores

# --- Setup ---
st.set_page_config(
    page_title="Ad Revenue Dashboard 🌐",
    layout="centered",
    page_icon="🌐",)
st.sidebar.title("Ad Revenue 🌐")
page = st.sidebar.selectbox("Select Page", [
                            "Introduction 📘", "Visualization 📊", "Prediction 🔮"])

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

    # st.header("📑 Automated Report")
    # if st.button("Generate Report"):
    #     with st.spinner("Generating report..."):
    #        profile = ProfileReport(
    #             df, title="Ads Revenue Report", explorative=True, minimal=True)
    #     st_profile_report(profile)

    #     export = profile.to_html()
    #     st.download_button(label="📥 Download full Report", data=export,
    #                       file_name="ads_revenue_report.html", mime='text/html')

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

# --- Prediction Page (TO BE WORKED ON) ---
elif page == "Prediction 🔮":
    st.title("03 Prediction (with Linear Regression) 🔮")
    st.markdown(""" ### What are we predicting ? Using campaign inputs (impressions, clicks, ad spend), we train a Linear Regression model to predict revenue. """)
    # --- Data Preprocessing ---

    # Remove missing values

    df = df.dropna()

    # Define X and Y
    features_selection = st.sidebar.multiselect(
        "Select features (X)", ["impressions", "clicks", "ad_spend"], default=["impressions", "clicks", "ad_spend"])
    target = "revenue"
    #makes sure something is selected
    if not features_selection:
        st.warning("Please select AT LEAST ONE feature from the sidebar.")
        st.stop()

    X = df[features_selection]
    y = df[target]

    selected_metrics = st.sidebar.multiselect("Metrics to display", [
                                              "Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R² Score"], default=["Mean Absolute Error (MAE)"])

    # st.dataframe(X.head())
    # st.dataframe(y.head())

    # split into train and test set
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_size = st.sidebar.slider("Training set size", 0.50, 0.95, 0.80, step=0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=42)

    # --- Model ---

    # Definition model
    model = LinearRegression()

    # Training model
    model.fit(X_train, y_train)

    # Prediction
    predictions = model.predict(X_test)

    # Evaluation
    
    st.header("📊 Model Performance")
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Training rows", f"{len(X_train):,}")
    col_info2.metric("Test rows", f"{len(X_test):,}")
    col_info3.metric("Features used", len(features_selection))
    st.markdown(" ")

    mae = metrics.mean_absolute_error(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    r2  = metrics.r2_score(y_test, predictions)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"${mae:,.2f}")
    col2.metric("MSE", f"${mse:,.2f}")
    col3.metric("R² Score", f"{r2:.3f}")

    if r2 >= 0.85:
        st.success(f" Strong model — explains {r2*100:.1f}% of revenue variance")
    elif r2 >= 0.60:
        st.info(f"ℹ Decent model — explains {r2*100:.1f}% of revenue variance")
    else:
        st.warning(f" Weak model — only explains {r2*100:.1f}% of revenue variance")

    st.markdown("---")

    st.header("Actual vs. Predicted Revenue")
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()], "--r", linewidth=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

    st.markdown("---")
    st.header("🏆 Which input drives revenue the most?")
    st.markdown("Each bar shows how much that variable influences predicted revenue.")

    coef_df = pd.DataFrame({
        "Feature": features_selection,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", ascending=False)

    fig_coef, ax_coef = plt.subplots(figsize=(8, 3))
    colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in coef_df["Coefficient"]]
    ax_coef.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    ax_coef.axvline(0, color="black", linewidth=0.8)
    ax_coef.set_xlabel("Effect on Revenue per 1-unit increase")
    ax_coef.set_title("Feature Coefficients (green = positive, red = negative)")
    st.pyplot(fig_coef)
    st.caption(f"Model baseline revenue (intercept): ${model.intercept_:,.2f}")
    
    st.markdown("---")
  

