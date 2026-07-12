import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("dataset/house_price_dataset_4000.csv")

# -----------------------------
# Title
# -----------------------------
st.title("🏠 House Price Prediction System")

st.write(
    "This application predicts house prices using Machine Learning."
)

# -----------------------------
# Sidebar
# -----------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📂 Dataset",
        "📊 Visualization",
        "🤖 Model Comparison",
        "🏡 Prediction",
        "ℹ️ About"
    ]
)

# ======================================================
# HOME PAGE
# ======================================================

if page == "🏠 Home":

    st.header("Welcome")
    st.image(
    "Images/house_photo.png",
    use_container_width=True
    )

    st.write("""
This project predicts house prices using Machine Learning.

### Features
- Data Cleaning
- Data Visualization
- Model Comparison
- House Price Prediction
- Streamlit Dashboard
""")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset", "4000 Rows")

    with col2:
        st.metric("Models", "3")

    with col3:
        st.metric("Best Model", "Linear Regression")


    st.success("✅ Best Model : Linear Regression (R² Score = 99.61%)")

    st.divider()

    st.subheader("Project Overview")

    st.write("""
    This Streamlit application predicts house prices using Machine Learning.

    The project includes:
    - Data Exploration
    - Data Visualization
    - Model Comparison
    - House Price Prediction
    """)

# ======================================================
# DATASET PAGE
# ======================================================

elif page == "📂 Dataset":

    st.header("Dataset Information")

    st.subheader("First 5 Rows")
    st.dataframe(data.head())

    st.subheader("Dataset Shape")
    st.write(data.shape)

    st.subheader("Column Names")
    st.write(list(data.columns))

    st.subheader("Data Types")
    st.write(data.dtypes)

    st.subheader("Missing Values")
    st.write(data.isnull().sum())
    st.subheader("Unique Values")

    st.subheader("Unique Values")

    st.write("Location:", data["Location"].unique())
    st.write("Furnishing:", data["Furnishing"].unique())
    st.write("Main_Road:", data["Main_Road"].unique())
    st.write("Air_Conditioning:", data["Air_Conditioning"].unique())
    st.write("Basement:", data["Basement"].unique())
    st.write("Garden:", data["Garden"].unique())

# ======================================================
# VISUALIZATION PAGE
# ======================================================

elif page == "📊 Visualization":

    st.header("Data Visualization")

    # ---------------- Area Distribution ----------------

    st.subheader("Area Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(
        data["Area_sqft"],
        bins=20,
        color="skyblue",
        edgecolor="black"
    )

    ax.set_title("Area Distribution")
    ax.set_xlabel("Area (sqft)")
    ax.set_ylabel("Number of Houses")

    st.pyplot(fig)

    # ---------------- House Price Distribution ----------------

    st.subheader("House Price Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(
        data["Price"],
        bins=20,
        color="orange",
        edgecolor="black"
    )

    ax.set_title("House Price Distribution")
    ax.set_xlabel("Price")
    ax.set_ylabel("Number of Houses")

    st.pyplot(fig)

    # ---------------- Average Price by Location ----------------

    st.subheader("Average Price by Location")

    avg_price = data.groupby("Location")["Price"].mean()

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(
        avg_price.index,
        avg_price.values,
        color="green"
    )

    ax.set_title("Average Price by Location")
    ax.set_xlabel("Location")
    ax.set_ylabel("Average Price")

    plt.xticks(rotation=45)

    st.pyplot(fig)


    # ======================================================
# MODEL COMPARISON PAGE
# ======================================================

elif page == "🤖 Model Comparison":

    st.header("🤖 Model Comparison")

    results = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "MAE": [
            311106.75,
            580472.85,
            380825.54
        ],
        "RMSE": [
            378496.35,
            722530.01,
            473624.89
        ],
        "R2 Score (%)": [
            99.61,
            98.56,
            99.33
        ]
    })

    st.subheader("Comparison Table")
    st.dataframe(results, use_container_width=True)

    # ---------------- R2 Score ----------------

    st.subheader("R2 Score Comparison")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(
        results["Model"],
        results["R2 Score (%)"],
        color=["green", "orange", "blue"]
    )

    ax.set_title("Model Comparison (R2 Score)")
    ax.set_xlabel("Models")
    ax.set_ylabel("R2 Score (%)")

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    st.pyplot(fig)

    # ---------------- RMSE ----------------

    st.subheader("RMSE Comparison")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(
        results["Model"],
        results["RMSE"],
        color=["purple", "red", "cyan"]
    )

    ax.set_title("RMSE Comparison")
    ax.set_xlabel("Models")
    ax.set_ylabel("RMSE")

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    st.pyplot(fig)






    # ======================================================
# PREDICTION PAGE
# ======================================================

elif page == "🏡 Prediction":

    st.header("🏡 House Price Prediction")

    # Load Model
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoder = joblib.load("models/encoder.pkl")


    house_id = st.number_input(
    "House ID",
    min_value=1,
    value=1
)

    area = st.number_input(
    "Area (sqft)",
    min_value=600,
    max_value=5000,
    value=1500
    )

    bedrooms = st.slider(
    "Bedrooms",
    1,
    6,
    3
    )

    bathrooms = st.slider(
    "Bathrooms",
    1,
    6,
    2
    )

    floors = st.slider(
    "Floors",
    1,
    3,
    1
    )

    parking = st.slider(
    "Parking",
    0,
    3,
    1
    )

    house_age = st.slider(
    "House Age",
    0,
    50,
    10
    )

    location = st.selectbox(
        "Location",
        [
            "Riverside",
            "Uptown",
            "Hill View",
            "City Center",
            "East Side",
            "Lake View",
            "West End",
            "Green Park",
            "Downtown",
            "Suburban"
        ]
    )
   

    furnishing = st.selectbox(
    "Furnishing",
    ["Furnished", "Semi-Furnished", "Unfurnished"]
    )

    main_road = st.selectbox(
    "Main Road",
    ["Yes", "No"]
    )

    air = st.selectbox(
    "Air Conditioning",
    ["Yes", "No"]
    )

    basement = st.selectbox(
    "Basement",
    ["Yes", "No"]
    )

    garden = st.selectbox(
    "Garden",
    ["Yes", "No"]
    )


    if st.button("Predict Price"):

        input_data = pd.DataFrame({
            "House_ID": [house_id],
            "Area_sqft": [area],
            "Bedrooms": [bedrooms],
            "Bathrooms": [bathrooms],
            "Floors": [floors],
            "Parking": [parking],
            "House_Age": [house_age],
            "Location": [location],
            "Furnishing": [furnishing],
            "Main_Road": [main_road],
            "Air_Conditioning": [air],
            "Basement": [basement],
            "Garden": [garden]
        })


            # Encode categorical columns
        categorical_cols = [
            "Location",
            "Furnishing",
            "Main_Road",
            "Air_Conditioning",
            "Basement",
            "Garden"
        ]

        input_data[categorical_cols] = encoder.transform(
            input_data[categorical_cols]
        )

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)

        st.success(
        f"🏠 Predicted House Price\n\n₹ {prediction[0]:,.0f}"
        ) 
        crore = prediction[0] / 10000000

        st.info(f"💰 Approx Price : {crore:.2f} Crore")

        st.snow()

# ======================================================
# ABOUT PAGE
# ======================================================

elif page == "ℹ️ About":

    st.header("ℹ️ About Project")

    st.write("""
## 🏠 House Price Prediction System

This project predicts house prices using Machine Learning.

The application allows users to:

- View Dataset
- Explore Data Visualization
- Compare Machine Learning Models
- Predict House Prices
- Use an Interactive Streamlit Dashboard
    """)

    st.subheader("📊 Dataset")

    st.write("""
- Dataset Size : 4000 Houses
- Features : 13
- Target : Price
    """)

    st.subheader("🤖 Machine Learning Models")

    st.write("""
- Linear Regression
- Decision Tree
- Random Forest
    """)

    st.subheader("📚 Python Libraries")

    st.write("""
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Joblib
    """)

    st.subheader("👨‍💻 Developed By")

    st.write("Aryan Guleria")
    