import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# 🎨 YlOrBr Color Palette
ylorbr_colors = [
    "#FFFFE5", "#FFF7BC", "#FEE391", "#FEC44F", "#FE9929", "#EC7014", "#CC4C02", "#993404", "#3e1f0d"
]

# 💅 Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(to bottom right, #FFF7BC, #FEC44F);
    }

    .result-box {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        border: 2px solid #3e1f0d;
        text-align: center;
        font-weight: bold;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }

    .stButton>button {
        background-color: #3e1f0d;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color:white;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# 🏷️ Title + Description inside a white card
st.markdown(f"""
    <div style='
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 30px;
        text-align: center;
    '>
      <span style='color:#3e1f0d; font-size:22px; font-weight:bold;'>🎯 Kickstarter Success Predictor</span>
      <p style='color:#000000; font-size:16px; margin: 0;'>Enter project details to predict pledged amount and success chance.</p>
    </div>
""", unsafe_allow_html=True)

# 📦 Load saved models and encoders
usd_model = joblib.load("usd_pledged_real_model.pkl")
class_model = joblib.load("decision_tree_model.pkl")
scaler_reg = joblib.load("scaler_regression.pkl")
scaler_class = joblib.load("scaler_classification.pkl")
encoder = joblib.load("onehot_encoder.pkl")
kmeans = joblib.load("kmeans_cluster.pkl")
regression_columns = joblib.load("regression_columns.pkl")

# 📝 Inputs (no white box)
usd_goal_real = st.number_input("🎯 USD Goal", min_value=100.0, max_value=1000000.0, value=5000.0, step=100.0)
backers = st.number_input("👥 Number of Backers", min_value=0, max_value=100000, value=100)
launched = st.date_input("🚀 Launched Date", value=datetime(2020, 1, 1))
deadline = st.date_input("⏳ Deadline Date", value=datetime(2020, 1, 31))

main_category = st.selectbox("📂 Main Category", [
    'Publishing', 'Film & Video', 'Music', 'Food', 'Crafts', 'Games', 'Design',
    'Comics', 'Fashion', 'Theater', 'Art', 'Photography', 'Technology', 'Dance', 'Journalism'
])

country = st.selectbox("🌍 Country", [
    'GB', 'US', 'CA', 'AU', 'NO', 'IT', 'DE', 'IE', 'MX', 'ES', 'SE', 'FR', 'NZ', 'CH',
    'AT', 'BE', 'DK', 'HK', 'NL', 'LU', 'SG', 'N,0"', 'JP'
])

# 🎯 Predict Button
if st.button("🔮 Predict"):
    try:
        # Feature engineering
        duration_days = (deadline - launched).days
        log_goal = np.log1p(usd_goal_real)
        log_backers = np.log1p(backers)
        log_duration = np.log1p(duration_days)

        df_new = pd.DataFrame({
            'log_usd_goal_real': [log_goal],
            'log_backers': [log_backers],
            'log_duration(Days)': [log_duration],
            'main_category': [main_category],
            'country': [country]
        })

        cluster = kmeans.predict(df_new[['log_usd_goal_real', 'log_backers', 'log_duration(Days)']])
        df_new['cluster'] = cluster

        encoded_cat = encoder.transform(df_new[['cluster','main_category', 'country']])
        encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(['cluster','main_category', 'country']))
        X_reg = pd.concat([df_new.drop(columns=['cluster','main_category', 'country']).reset_index(drop=True), encoded_cat_df.reset_index(drop=True)], axis=1)

        for col in regression_columns:
            if col not in X_reg.columns:
                X_reg[col] = 0
        X_reg = X_reg[regression_columns]

        X_reg_scaled = scaler_reg.transform(X_reg)
        pred_log_usd = usd_model.predict(X_reg_scaled)
        pred_usd = np.expm1(pred_log_usd)

        X_cls_input = np.hstack((X_reg_scaled, pred_log_usd.reshape(-1, 1)))
        X_cls_scaled = scaler_class.transform(X_cls_input)
        pred_state = class_model.predict(X_cls_scaled)
        state_text = "✅ Will Succeed" if pred_state[0] == "successful" else "❌ Will Fail"

        st.markdown(f"""
            <div class='result-box'>
                📊 Predicted USD Pledged:
                <span style='color:#3e1f0d; font-size:20px;'>${pred_usd[0]:,.2f}</span><br>
                {state_text}
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error in prediction: {e}")
