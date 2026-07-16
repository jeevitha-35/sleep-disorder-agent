# ==========================================================
# Sleep Disorder Prediction AI Agent
# Part-1 : Imports, Model Loading & User Inputs
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="AI Sleep Health Assistant",
    page_icon="😴",
    layout="wide"
)

# ----------------------------------------------------------
# Title
# ----------------------------------------------------------

st.title("😴 AI Sleep Health Assistant")
st.markdown("""
Predict **Sleep Disorders** using Machine Learning and receive
personalized health recommendations, precautions, and wellness guidance.
""")

st.markdown("---")

# ----------------------------------------------------------
# Load Model
# ----------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("sleep_model.pkl")

@st.cache_resource
def load_encoders():
    return joblib.load("encoders.pkl")

@st.cache_resource
def load_features():
    return joblib.load("feature_names.pkl")

model = load_model()
encoders = load_encoders()
feature_names = load_features()

# ----------------------------------------------------------
# User Inputs
# ----------------------------------------------------------

st.header("📝 Enter Your Details")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        encoders["Gender"].classes_
    )

    age = st.number_input(
        "Age",
        min_value=10,
        max_value=100,
        value=25
    )

    occupation = st.selectbox(
        "Occupation",
        encoders["Occupation"].classes_
    )

    sleep_duration = st.slider(
        "Sleep Duration (Hours)",
        0.0,
        12.0,
        7.0,
        0.1
    )

    quality_of_sleep = st.slider(
        "Quality of Sleep",
        1,
        10,
        7
    )

    physical_activity = st.slider(
        "Physical Activity Level",
        0,
        100,
        50
    )

with col2:

    stress_level = st.slider(
        "Stress Level",
        1,
        10,
        5
    )

    bmi_category = st.selectbox(
        "BMI Category",
        encoders["BMI Category"].classes_
    )

    systolic_bp = st.number_input(
        "Systolic BP",
        min_value=70,
        max_value=250,
        value=120
    )

    diastolic_bp = st.number_input(
        "Diastolic BP",
        min_value=40,
        max_value=150,
        value=80
    )

    heart_rate = st.number_input(
        "Heart Rate",
        min_value=30,
        max_value=200,
        value=72
    )

    daily_steps = st.number_input(
        "Daily Steps",
        min_value=0,
        max_value=50000,
        value=7000
    )

st.markdown("---")

# ----------------------------------------------------------
# Predict Button
# ----------------------------------------------------------

predict_button = st.button(
    "🔍 Predict Sleep Disorder",
    use_container_width=True
)

# ----------------------------------------------------------
# Prepare Input Data
# ----------------------------------------------------------

input_data = {
    "Gender": gender,
    "Age": age,
    "Occupation": occupation,
    "Sleep Duration": sleep_duration,
    "Quality of Sleep": quality_of_sleep,
    "Physical Activity Level": physical_activity,
    "Stress Level": stress_level,
    "BMI Category": bmi_category,
    "Heart Rate": heart_rate,
    "Daily Steps": daily_steps,
    "Systolic BP": systolic_bp,
    "Diastolic BP": diastolic_bp
}

input_df = pd.DataFrame([input_data])
# ==========================================================
# Part-2 : Data Encoding, Prediction & Risk Analysis
# ==========================================================

if predict_button:

    # ------------------------------------------------------
    # Encode Categorical Features
    # ------------------------------------------------------

    input_df["Gender"] = encoders["Gender"].transform(
        input_df["Gender"]
    )

    input_df["Occupation"] = encoders["Occupation"].transform(
        input_df["Occupation"]
    )

    input_df["BMI Category"] = encoders["BMI Category"].transform(
        input_df["BMI Category"]
    )

    # ------------------------------------------------------
    # Arrange Features
    # ------------------------------------------------------

    input_df = input_df[feature_names]

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    prediction = model.predict(input_df)[0]

    predicted_sleep_disorder = encoders["Sleep Disorder"].inverse_transform(
        [prediction]
    )[0]

    # ------------------------------------------------------
    # Prediction Probability
    # ------------------------------------------------------

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        confidence = np.max(probabilities) * 100

    # ------------------------------------------------------
    # Display Prediction
    # ------------------------------------------------------

    st.markdown("---")

    st.header("🧠 AI Prediction")

    if predicted_sleep_disorder == "None":

        st.success("✅ No Sleep Disorder Detected")

    elif predicted_sleep_disorder == "Insomnia":

        st.warning("😴 Predicted Sleep Disorder : Insomnia")

    else:

        st.error(f"⚠ Predicted Sleep Disorder : {predicted_sleep_disorder}")

    # ------------------------------------------------------
    # Confidence Score
    # ------------------------------------------------------

    if confidence is not None:

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

    # ------------------------------------------------------
    # Risk Level
    # ------------------------------------------------------

    risk_score = 0

    if sleep_duration < 6:
        risk_score += 2

    elif sleep_duration < 7:
        risk_score += 1

    if quality_of_sleep <= 4:
        risk_score += 2

    elif quality_of_sleep <= 6:
        risk_score += 1

    if stress_level >= 8:
        risk_score += 2

    elif stress_level >= 6:
        risk_score += 1

    if daily_steps < 5000:
        risk_score += 2

    elif daily_steps < 8000:
        risk_score += 1

    if heart_rate > 100:
        risk_score += 2

    if systolic_bp >= 140 or diastolic_bp >= 90:
        risk_score += 2

    elif systolic_bp >= 130 or diastolic_bp >= 80:
        risk_score += 1

    # ------------------------------------------------------
    # Display Risk
    # ------------------------------------------------------

    st.subheader("📊 Overall Risk Level")

    if risk_score <= 2:

        st.success("🟢 LOW RISK")

    elif risk_score <= 5:

        st.warning("🟡 MODERATE RISK")

    else:

        st.error("🔴 HIGH RISK")

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    st.markdown("### 📋 Health Summary")

    st.write(f"**Age :** {age}")

    st.write(f"**Gender :** {gender}")

    st.write(f"**Occupation :** {occupation}")

    st.write(f"**Sleep Duration :** {sleep_duration} Hours")

    st.write(f"**Quality of Sleep :** {quality_of_sleep}/10")

    st.write(f"**Stress Level :** {stress_level}/10")

    st.write(f"**Physical Activity :** {physical_activity}")

    st.write(f"**BMI Category :** {bmi_category}")

    st.write(f"**Blood Pressure :** {systolic_bp}/{diastolic_bp}")

    st.write(f"**Heart Rate :** {heart_rate} bpm")

    st.write(f"**Daily Steps :** {daily_steps}")

    # ------------------------------------------------------
    # AI Explanation
    # ------------------------------------------------------

    st.markdown("---")
    st.subheader("🤖 AI Explanation")

    reasons = []

    if sleep_duration < 7:
        reasons.append(
            "• Your sleep duration is below the recommended 7–9 hours."
        )

    if quality_of_sleep <= 5:
        reasons.append(
            "• Your sleep quality score is low."
        )

    if stress_level >= 7:
        reasons.append(
            "• High stress may negatively affect sleep."
        )

    if daily_steps < 6000:
        reasons.append(
            "• Low daily physical activity can reduce sleep quality."
        )

    if systolic_bp >= 140 or diastolic_bp >= 90:
        reasons.append(
            "• Elevated blood pressure may be associated with poor sleep."
        )

    if heart_rate > 100:
        reasons.append(
            "• Elevated resting heart rate may indicate stress or other health concerns."
        )

    if len(reasons) == 0:

        st.success(
            "✅ Your lifestyle indicators appear generally healthy based on the provided information."
        )

    else:

        st.info(
            "The prediction may have been influenced by the following factors:"
        )

        for reason in reasons:
            st.write(reason)

    st.markdown("---")
    # ==========================================================
# Part-3A : AI Health Assistant
# Personalized Precautions & Health Recommendations
# ==========================================================

    st.markdown("---")
    st.header("🩺 AI Health Assistant")

    # ------------------------------------------------------
    # AI Health Score
    # ------------------------------------------------------

    health_score = 100

    if sleep_duration < 7:
        health_score -= 15

    if quality_of_sleep <= 5:
        health_score -= 15

    if stress_level >= 7:
        health_score -= 15

    if daily_steps < 6000:
        health_score -= 10

    if systolic_bp >= 140 or diastolic_bp >= 90:
        health_score -= 15

    if heart_rate > 100:
        health_score -= 10

    if health_score < 0:
        health_score = 0

    st.subheader("💚 AI Health Score")

    st.progress(health_score / 100)

    st.metric(
        "Overall Health Score",
        f"{health_score}/100"
    )

    # ------------------------------------------------------
    # Health Rating
    # ------------------------------------------------------

    if health_score >= 90:
        st.success("⭐⭐⭐⭐⭐ Excellent Health")

    elif health_score >= 75:
        st.success("⭐⭐⭐⭐ Very Good")

    elif health_score >= 60:
        st.warning("⭐⭐⭐ Good but Needs Improvement")

    elif health_score >= 40:
        st.warning("⭐⭐ Fair")

    else:
        st.error("⭐ Poor - Lifestyle Changes Recommended")

    # ------------------------------------------------------
    # Personalized Precautions
    # ------------------------------------------------------

    st.markdown("---")
    st.subheader("🛡 Personalized Precautions")

    precautions = []

    if predicted_sleep_disorder == "Insomnia":

        precautions.extend([
            "Maintain a fixed bedtime every day.",
            "Avoid using mobile phones before sleeping.",
            "Avoid caffeine after evening.",
            "Do not consume heavy meals before bedtime.",
            "Practice deep breathing before sleep.",
            "Avoid daytime naps longer than 30 minutes."
        ])

    elif predicted_sleep_disorder == "Sleep Apnea":

        precautions.extend([
            "Maintain a healthy body weight.",
            "Sleep on your side instead of your back.",
            "Avoid alcohol before bedtime.",
            "Avoid smoking.",
            "Consult a sleep specialist.",
            "Follow medical advice if CPAP therapy is recommended."
        ])

    else:

        precautions.extend([
            "Continue maintaining healthy sleep habits.",
            "Sleep 7–9 hours daily.",
            "Exercise regularly.",
            "Maintain a balanced diet.",
            "Manage stress effectively."
        ])

    for item in precautions:
        st.write("✅", item)

    # ------------------------------------------------------
    # Health Warnings
    # ------------------------------------------------------

    st.markdown("---")
    st.subheader("⚠ Health Warnings")

    if systolic_bp >= 140 or diastolic_bp >= 90:
        st.error("High Blood Pressure detected.")
        st.write("• Monitor your blood pressure regularly.")
        st.write("• Reduce salt intake.")
        st.write("• Consult a healthcare professional if readings remain elevated.")

    if stress_level >= 8:
        st.error("High Stress Level detected.")
        st.write("• Practice meditation.")
        st.write("• Try yoga.")
        st.write("• Maintain work-life balance.")
        st.write("• Sleep adequately.")

    if daily_steps < 5000:
        st.warning("Low Physical Activity detected.")
        st.write("• Aim for at least 8,000 steps daily.")
        st.write("• Walk for 30 minutes.")
        st.write("• Take short movement breaks.")

    if heart_rate > 100:
        st.warning("High Resting Heart Rate detected.")
        st.write("• Reduce stress.")
        st.write("• Stay hydrated.")
        st.write("• Consult a healthcare professional if persistent.")

    if sleep_duration < 6:
        st.warning("Insufficient Sleep Duration.")
        st.write("• Adults generally benefit from 7–9 hours of sleep per night.")
        st.write("• Establish a consistent bedtime routine.")

    # ------------------------------------------------------
    # Sleep Hygiene Tips
    # ------------------------------------------------------

    st.markdown("---")
    st.subheader("🌙 Sleep Hygiene Tips")

    sleep_tips = [
        "Sleep at the same time every day.",
        "Wake up at the same time every morning.",
        "Avoid mobile phones one hour before bedtime.",
        "Reduce room light before sleeping.",
        "Keep your bedroom quiet and cool.",
        "Avoid large meals before bedtime.",
        "Limit caffeine in the evening.",
        "Avoid smoking before sleeping.",
        "Exercise regularly, but not immediately before bed.",
        "Read a book instead of using electronic devices."
    ]

    for tip in sleep_tips:
        st.write("🌙", tip)

    # ------------------------------------------------------
    # Lifestyle Recommendations
    # ------------------------------------------------------

    st.markdown("---")
    st.subheader("🏃 Lifestyle Recommendations")

    st.write("💧 Drink approximately 2–3 liters of water daily, unless advised otherwise by a healthcare professional.")
    st.write("🚶 Aim for 8,000–10,000 steps per day.")
    st.write("🏋 Exercise for at least 30 minutes on most days of the week.")
    st.write("🥗 Eat more fruits, vegetables and whole grains.")
    st.write("🍔 Reduce fast food and sugary beverages.")
    st.write("😌 Practice relaxation techniques daily.")
    st.write("📵 Limit screen time before sleeping.")
    st.write("🌞 Get regular morning sunlight exposure.")

    # ------------------------------------------------------
    # Positive Motivation
    # ------------------------------------------------------

    st.markdown("---")
    st.subheader("💙 Daily Wellness Reminder")

    st.info(
        "Small healthy habits practiced consistently can improve sleep quality and overall well-being over time."
    )

    st.markdown("---")
    # ==========================================================
# Part-3B : Diet Plan, Exercise Plan,
# Medical Tests, Emergency Symptoms,
# Weekly Sleep Improvement Plan
# ==========================================================

    st.markdown("---")
    st.header("🥗 Personalized Diet Plan")

    if predicted_sleep_disorder == "Insomnia":

        st.success("Foods That May Support Better Sleep")

        foods = [
            "Warm Milk",
            "Banana",
            "Almonds",
            "Walnuts",
            "Oats",
            "Kiwi",
            "Chamomile Tea",
            "Brown Rice",
            "Leafy Green Vegetables",
            "Yogurt"
        ]

        avoid = [
            "Coffee",
            "Energy Drinks",
            "Alcohol",
            "Sugary Foods",
            "Spicy Foods",
            "Heavy Dinner"
        ]

    elif predicted_sleep_disorder == "Sleep Apnea":

        st.success("Healthy Weight Management Diet")

        foods = [
            "Vegetables",
            "Fruits",
            "Whole Grains",
            "Fish",
            "Lean Chicken",
            "Beans",
            "Low-fat Milk",
            "Salads",
            "Nuts",
            "Seeds"
        ]

        avoid = [
            "Fast Food",
            "Deep Fried Foods",
            "Sugary Drinks",
            "Alcohol",
            "Processed Meat",
            "Excess Salt"
        ]

    else:

        st.success("Healthy Balanced Diet")

        foods = [
            "Fresh Fruits",
            "Vegetables",
            "Whole Grains",
            "Lean Protein",
            "Eggs",
            "Milk",
            "Nuts",
            "Seeds",
            "Fish",
            "Beans"
        ]

        avoid = [
            "Junk Food",
            "Soft Drinks",
            "Too Much Sugar",
            "Excess Salt",
            "Alcohol"
        ]

    st.subheader("✅ Recommended Foods")

    for food in foods:
        st.write("🥦", food)

    st.subheader("❌ Limit These Foods")

    for item in avoid:
        st.write("🚫", item)

    # ------------------------------------------------------
    # Exercise Plan
    # ------------------------------------------------------

    st.markdown("---")
    st.header("🏋 Personalized Exercise Plan")

    if daily_steps < 5000:

        st.warning("Your daily activity is low.")

        exercise = [
            "Walk 30 minutes daily",
            "Morning Stretching",
            "Cycling",
            "Light Jogging",
            "Use Stairs Instead of Lift"
        ]

    else:

        exercise = [
            "30–45 minutes Walking",
            "Strength Training (3 Days/Week)",
            "Yoga",
            "Stretching",
            "Weekend Outdoor Activities"
        ]

    for ex in exercise:
        st.write("🏃", ex)

    # ------------------------------------------------------
    # Stress Management
    # ------------------------------------------------------

    st.markdown("---")
    st.header("🧘 Stress Management")

    stress_tips = [
        "Meditation (10-15 minutes daily)",
        "Deep Breathing Exercises",
        "Yoga",
        "Listen to Relaxing Music",
        "Read Books",
        "Spend Time in Nature",
        "Talk with Friends or Family",
        "Practice Gratitude",
        "Maintain Work-Life Balance",
        "Avoid Overthinking Before Bed"
    ]

    for tip in stress_tips:
        st.write("💙", tip)

    # ------------------------------------------------------
    # Medical Tests
    # ------------------------------------------------------

    st.markdown("---")
    st.header("🩺 Recommended Medical Tests")

    if predicted_sleep_disorder == "Sleep Apnea":

        tests = [
            "Sleep Study (Polysomnography)",
            "Overnight Pulse Oximetry",
            "ENT Examination",
            "BMI Assessment",
            "Blood Pressure Check"
        ]

    elif predicted_sleep_disorder == "Insomnia":

        tests = [
            "Sleep Diary",
            "Mental Health Assessment",
            "Stress Evaluation",
            "Physical Examination",
            "Thyroid Function Test (if advised)"
        ]

    else:

        tests = [
            "Annual Health Check-up",
            "Blood Pressure Monitoring",
            "Blood Sugar Test",
            "BMI Assessment"
        ]

    for test in tests:
        st.write("✔", test)

    # ------------------------------------------------------
    # Emergency Symptoms
    # ------------------------------------------------------

    st.markdown("---")
    st.header("🚨 Emergency Warning Signs")

    emergency = [
        "Stopping breathing during sleep",
        "Severe chest pain",
        "Sudden fainting",
        "Extreme daytime sleepiness while driving",
        "Confusion or loss of consciousness",
        "Persistent high blood pressure with symptoms"
    ]

    for sign in emergency:
        st.write("⚠", sign)

    st.error(
        "If you experience any of the above symptoms, seek immediate medical care."
    )

    # ------------------------------------------------------
    # Weekly Sleep Improvement Plan
    # ------------------------------------------------------

    st.markdown("---")
    st.header("📅 4-Week Sleep Improvement Plan")

    week1 = [
        "Sleep before 10:30 PM",
        "Wake up at the same time daily",
        "Avoid caffeine in the evening"
    ]

    week2 = [
        "Walk 30 minutes every day",
        "Drink enough water",
        "Reduce screen time before bed"
    ]

    week3 = [
        "Practice meditation daily",
        "Eat healthy meals",
        "Avoid heavy dinner"
    ]

    week4 = [
        "Maintain 7–9 hours of sleep",
        "Continue exercise routine",
        "Monitor stress levels"
    ]

    st.subheader("Week 1")
    for item in week1:
        st.write("✅", item)

    st.subheader("Week 2")
    for item in week2:
        st.write("✅", item)

    st.subheader("Week 3")
    for item in week3:
        st.write("✅", item)

    st.subheader("Week 4")
    for item in week4:
        st.write("✅", item)

    # ------------------------------------------------------
    # Final AI Summary
    # ------------------------------------------------------

    st.markdown("---")
    st.header("🤖 AI Wellness Summary")

    st.success(f"""
Prediction : {predicted_sleep_disorder}

Health Score : {health_score}/100

✔ Follow the personalized precautions.

✔ Maintain healthy sleep habits.

✔ Exercise regularly.

✔ Eat a balanced diet.

✔ Manage stress effectively.

✔ Seek medical advice if symptoms persist or worsen.
""")

    st.info(
        "⚠ This application is for educational purposes only and is not a substitute for professional medical diagnosis or treatment."
    )

    st.markdown("---")
    