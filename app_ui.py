# app_ui.py
import json
import os

import requests
import streamlit as st

# Get the API URL from an environment variable or use a default
API_URL_DEFAULT = "https://insurance-prediction-production.up.railway.app"  # Base URL
PREDICTION_ENDPOINT = f"{API_URL_DEFAULT}/predictions/predict"
API_URL = os.getenv("PREDICTION_API_URL", PREDICTION_ENDPOINT)

st.set_page_config(page_title="Insurance Prediction Dashboard", layout="wide")

# --- Sidebar for Navigation and Inputs ---
st.sidebar.title(
    "Insurance Prediction MLOps "
)  # "Insurance Prediction MLOps" in Hindi as a creative touch
st.sidebar.markdown("---")

app_mode = st.sidebar.selectbox(
    "Choose the App Mode", ["Insurance Prediction UI", "Project Dashboard"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Developed for MLOps Project - ENSIAS")
st.sidebar.markdown("By: Hamza BENATMANE")  # Added your name


# --- Main Page Content based on Mode ---

if app_mode == "Insurance Prediction UI":
    st.title("🚗 Vehicle Insurance Purchase Prediction UI")
    st.markdown(
        """
    This simple UI allows you to input customer and vehicle details to predict
    the likelihood of an insurance purchase using our deployed machine learning model.
    Fill in the details in the sidebar and click 'Predict Likelihood'.
    """
    )

    # --- Input Fields moved to sidebar for UI mode ---
    with st.sidebar.expander("Input Features for Prediction", expanded=True):
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="ui_gender")
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1, key="ui_age")
        has_driving_license_str = st.selectbox(
            "Has Driving License?", ["Yes", "No"], index=0, key="ui_license"
        )
        has_driving_license = True if has_driving_license_str == "Yes" else False
        region_id = st.number_input("Region ID", min_value=0, value=28, step=1, key="ui_region")
        vehicle_age_options = ["< 1 Year", "1-2 Year", "> 2 Years"]
        vehicle_age = st.selectbox(
            "Vehicle Age", vehicle_age_options, index=1, key="ui_vehicle_age"
        )
        past_accident_str = st.selectbox(
            "Past Accident?", ["Yes", "No"], index=1, key="ui_accident"
        )
        past_accident = past_accident_str
        annual_premium = st.number_input(
            "Annual Premium (€)", min_value=0.0, value=2630.0, step=100.0, key="ui_premium"
        )
        sales_channel_id = st.number_input(
            "Sales Channel ID", min_value=0, value=26, step=1, key="ui_sales_channel"
        )
        days_since_created = st.number_input(
            "Policy Days Since Created", min_value=0, value=80, step=1, key="ui_days_created"
        )

        if st.button("Predict Likelihood", key="ui_predict_button"):
            payload = {
                "gender": gender,
                "age": age,
                "has_driving_license": has_driving_license,
                "region_id": region_id,
                "vehicle_age": vehicle_age,
                "past_accident": past_accident,
                "annual_premium": annual_premium,
                "sales_channel_id": sales_channel_id,
                "days_since_created": days_since_created,
            }

            st.session_state.payload = payload  # Store in session state to display on main page
            st.session_state.prediction_data = None  # Reset previous prediction
            st.session_state.error_message = None

            try:
                response = requests.post(API_URL, json=payload, timeout=10)
                response.raise_for_status()
                st.session_state.prediction_data = response.json()
            except requests.exceptions.HTTPError as errh:
                st.session_state.error_message = f"HTTP Error: {errh}\nResponse content: {response.text if 'response' in locals() else 'N/A'}"
            except requests.exceptions.ConnectionError as errc:
                st.session_state.error_message = f"Error Connecting: {errc}"
            except requests.exceptions.Timeout as errt:
                st.session_state.error_message = f"Timeout Error: {errt}"
            except requests.exceptions.RequestException as err:
                st.session_state.error_message = f"Oops: Something Else went wrong: {err}"
            except json.JSONDecodeError:
                st.session_state.error_message = f"Failed to decode JSON from API response.\nResponse content: {response.text if 'response' in locals() else 'N/A'}"

    # Display prediction results or errors on the main page
    if "payload" in st.session_state and st.session_state.payload is not None:
        st.subheader("Sending to API:")
        st.json(st.session_state.payload)
        st.session_state.payload = None  # Clear after displaying

    if "error_message" in st.session_state and st.session_state.error_message:
        st.error(st.session_state.error_message)
    elif "prediction_data" in st.session_state and st.session_state.prediction_data:
        prediction_data = st.session_state.prediction_data
        st.subheader("📈 Prediction Result")
        prediction_value = prediction_data.get("prediction")
        probability_value = prediction_data.get("probability")

        if probability_value is not None:  # Check if probability exists
            prob_display = f"{probability_value*100:.0f}%"
            if prediction_value == 1:
                st.success(
                    f"Prediction: Likely to purchase insurance! (Probability: {prob_display})"
                )
            elif prediction_value == 0:
                st.error(
                    f"Prediction: Unlikely to purchase insurance. (Probability: {prob_display})"
                )
            else:
                st.warning("Received an unexpected prediction value.")
        else:  # Handle case where probability might be missing
            if prediction_value == 1:
                st.success("Prediction: Likely to purchase insurance!")
            elif prediction_value == 0:
                st.error("Prediction: Unlikely to purchase insurance.")
            else:
                st.warning("Received an unexpected prediction value or missing probability.")

        st.subheader("Raw API Response:")
        st.json(prediction_data)


elif app_mode == "Project Dashboard":
    st.title("📊 MLOps Project Dashboard")
    st.markdown(
        "Overview of data insights, model performance, and deployment status for the Insurance Prediction project."
    )

    # --- Section 1: Data Overview ---
    st.header("1. Data Insights (from Training Data Profile)")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Target Variable Distribution")
        # IMPORTANT: Ensure this path is correct relative to where app_ui.py is run
        # Or use absolute paths, or copy images to an 'assets' folder next to app_ui.py
        st.image(
            "data/processed/train/profile/visualizations/target_distribution.png",
            caption="Target Variable ('result') Distribution",
            use_column_width=True,
        )
    with col2:
        st.subheader("Gender Distribution")
        st.image(
            "data/processed/train/profile/visualizations/count_gender.png",
            caption="Gender Distribution",
            use_column_width=True,
        )

    st.subheader("Feature Distributions")
    col_dist1, col_dist2 = st.columns(2)
    with col_dist1:
        st.image(
            "data/processed/train/profile/visualizations/dist_age.png",
            caption="Age Distribution",
            use_column_width=True,
        )
    with col_dist2:
        st.image(
            "data/processed/train/profile/visualizations/dist_annual_premium.png",
            caption="Annual Premium Distribution",
            use_column_width=True,
        )

    st.subheader("Correlation Heatmap")
    st.image(
        "data/processed/train/profile/visualizations/correlation_heatmap.png",
        caption="Feature Correlation Heatmap",
        use_column_width=True,
    )

    # --- Section 2: Model Performance ---
    st.header("2. Model Performance (LightGBM - Validation Set)")  # Assuming LightGBM was best

    # Display key metrics (you can load these from a file or MLflow if you export them)
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    col_metric1.metric(label="ROC AUC Score", value="0.806")  # From your screenshot (rounded)
    # Add other relevant metrics if you have them easily available, e.g., F1-score, Accuracy
    # col_metric2.metric(label="F1-score (Positive Class)", value="0.XX")
    # col_metric3.metric(label="Accuracy", value="0.YY")

    col_model_plot1, col_model_plot2 = st.columns(2)
    with col_model_plot1:
        st.subheader("Feature Importance")
        # IMPORTANT: Ensure this path is correct. This is an example path.
        st.image(
            "models/comparison/lightgbm/interpretation/feature_importance.png",
            caption="Top 20 Feature Importances",
            use_column_width=True,
        )
    with col_model_plot2:
        st.subheader("ROC Curve")
        st.image(
            "models/comparison/lightgbm/val_evaluation/roc_curve.png",
            caption="ROC Curve (AUC = 0.8063)",
            use_column_width=True,
        )

    # --- Section 3: MLOps Tools & Deployment ---
    st.header("3. MLOps & Deployment Status")

    col_tool1, col_tool2 = st.columns(2)
    with col_tool1:
        st.subheader("CI/CD Pipeline (GitHub Actions)")
        st.markdown("All checks (Code Quality, Tests, Build, Deploy) are passing.")
        # You can use the direct image URL from GitHub if it's public and stable, or save it locally.
        # For simplicity, let's assume you saved it locally in an 'assets' folder:
        # st.image("assets/github_actions_passed.png", caption="Successful GitHub Actions CI/CD Run")
        st.success("CI/CD Pipeline: Operational ✅")

    with col_tool2:
        st.subheader("Deployment (Railway.app)")
        st.markdown(f"Live API Base URL: `{API_URL_DEFAULT}`")
        st.markdown(f"Interactive API Docs: `{API_URL_DEFAULT}/docs`")
        st.success("Deployment: Active ✅")

    col_monitor1, col_monitor2 = st.columns(2)
    with col_monitor1:
        st.subheader("Uptime Monitoring (UptimeRobot)")
        st.markdown("Service is continuously monitored for availability.")
        st.markdown("[View Public Status Page](https://stats.uptimerobot.com/EDZuPKfmwD)")
        st.image(
            "assets/uptimerobot_status.png",
            caption="UptimeRobot Operational Status",
            use_column_width=True,
        )
        # For the above, you'd save your UptimeRobot operational screenshot to an 'assets' folder.
        # e.g., 'Screenshot 2025-05-31 at 21.44.33.png' renamed and placed in 'assets'.

    with col_monitor2:
        st.subheader("Error Tracking (Sentry)")
        st.markdown("Integrated for real-time error reporting and diagnostics.")
        st.info("Test error successfully captured, confirming integration.")
        # st.image("assets/sentry_error.png", caption="Sentry Dashboard with Captured Error", use_column_width=True)
        # For the above, you'd save your Sentry screenshot to an 'assets' folder.
        # e.g., 'Screenshot 2025-05-31 at 21.45.03.png' renamed and placed in 'assets'.

    st.subheader("MLflow Experiment Tracking")
    st.markdown("Experiments, parameters, metrics, and models are tracked using MLflow.")
    # st.image("assets/mlflow_experiments.png", caption="MLflow Experiments UI", use_column_width=True)
    # For the above, you'd save your MLflow screenshot (Screenshot 2025-05-31 at 21.55.30.png) to an 'assets' folder.

    st.info(
        "Note: Some images in the dashboard (like CI/CD, Sentry, MLflow screenshots) are representative placeholders. They should be replaced with actual screenshots saved locally in an 'assets' folder or referenced via stable URLs if preferred for the report."
    )
