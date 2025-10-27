# -------------------------------------------------------------
# 🌿 Carbon Footprint Estimator + Chatbot Suggestion Engine

# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------------------------------------
#  PAGE SETUP
# -------------------------------------------------------------
st.set_page_config(page_title="Carbon Footprint Assistant", page_icon="🌿", layout="wide")

st.title("🌿 Carbon Footprint Estimator & Sustainability Chatbot")
st.write("Estimate your CO₂ emissions and chat with an eco-assistant for green living tips 🌎")

# -------------------------------------------------------------
#  LOAD DATASET
# -------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Carbon Emission.csv")
    df = df.fillna(df.mode().iloc[0])  # fill missing values with mode
    if 'Emission' not in df.columns:
        df.rename(columns={df.columns[-1]: 'Emission'}, inplace=True)
    return df

df = load_data()
st.success("✅ Dataset Loaded Successfully!")

# -------------------------------------------------------------
#  ENCODE DATA FOR TRAINING
# -------------------------------------------------------------
encoded_df = df.copy()
label_encoders = {}

for col in encoded_df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    encoded_df[col] = le.fit_transform(encoded_df[col])
    label_encoders[col] = le

X = encoded_df.drop(columns=['Emission'])
y = encoded_df['Emission']

# -------------------------------------------------------------
#  TRAIN MODEL
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"📊 *Model Performance:* MAE = {mae:.2f}, R² = {r2:.2f}")

# -------------------------------------------------------------
# LAYOUT: TWO COLUMNS (Estimator + Chatbot)
# -------------------------------------------------------------
col1, col2 = st.columns(2)

# -------------------------------------------------------------
#  COLUMN 1: CARBON FOOTPRINT ESTIMATOR
# -------------------------------------------------------------
with col1:
    st.subheader("🧾 Estimate Your Carbon Footprint")

    user_inputs = {}

    # For every column in dataset except 'Emission'
    for col in X.columns:
        if df[col].dtype == 'object':
            options = sorted(df[col].unique().tolist())
            selected = st.selectbox(f"{col}", options)
            user_inputs[col] = selected
        else:
            val = st.number_input(f"{col}", min_value=0.0, step=0.1)
            user_inputs[col] = val

    if st.button("🌱 Estimate"):
        user_df = pd.DataFrame([user_inputs])

        # Display the user's alphabetic input
        st.write("### ✨ Your Inputs (Alphabetical):")
        st.table(user_df)

        # Encode user inputs same as training
        encoded_input = user_df.copy()
        for col in encoded_input.columns:
            if col in label_encoders:
                le = label_encoders[col]
                encoded_input[col] = le.transform(encoded_input[col])

        # Prediction
        predicted = model.predict(encoded_input)[0]
        st.success(f"Your Estimated Monthly Carbon Footprint: *{round(predicted, 2)} kg CO₂e*")

        # -------------------------------------------------------------
        # 🌱 Suggestion Engine
        # -------------------------------------------------------------
        def suggestion_engine(predicted):
            tips = []
            if predicted > 600:
                tips.append(" Try using public transport or reducing vehicle distance.")
                tips.append(" Switch to renewable energy sources where possible.")
            elif predicted > 400:
                tips.append(" Reduce long showers and unplug unused appliances.")
                tips.append(" Adopt a more plant-based diet.")
            else:
                tips.append(" Excellent! Your carbon footprint is within a sustainable range.")
            return tips

        tips = suggestion_engine(predicted)
        st.subheader(" Personalized Suggestions:")
        for tip in tips:
            st.write("- ", tip)

        # -------------------------------------------------------------
        #  Visualization
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(encoded_input.columns, encoded_input.iloc[0], color="seagreen")
        ax.set_title("Your Encoded Input Factors")
        ax.set_ylabel("Encoded Values")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# -------------------------------------------------------------
#  COLUMN 2: CHATBOT FOR GREEN ADVICE
# -------------------------------------------------------------
with col2:
    st.subheader(" Ask the Eco-Assistant")

    st.info("Ask me things like:\n• How can I reduce CO₂ from travel?\n• What are sustainable cooking methods?\n• How can I save electricity at home?")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Get user input for chat
    user_input = st.chat_input("Type your sustainability question here...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        q = user_input.lower()
        response = " I'm here to help! "

        if "electricity" in q:
            response += "Switch to LED bulbs, unplug unused devices, and install solar panels."
        elif "travel" in q or "transport" in q:
            response += "Use public transport, carpool, or bike for short trips."
        elif "cooking" in q:
            response += "Cook efficiently using LPG or induction stoves."
        elif "waste" in q:
            response += "Reduce, reuse, recycle, and compost organic waste."
        elif "diet" in q or "food" in q:
            response += "Eat more vegetables and reduce meat consumption."
        elif "water" in q:
            response += "Fix leaks, reuse water, and avoid long showers."
        elif "carbon footprint" in q:
            response += "Track your footprint regularly using this app and improve step by step."
        else:
            response += "Reduce energy, transport, and waste emissions for a greener lifestyle."

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
