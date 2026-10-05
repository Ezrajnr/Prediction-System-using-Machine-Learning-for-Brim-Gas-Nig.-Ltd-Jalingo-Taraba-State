import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Config
st.set_page_config(page_title="Brim Gas Sales Prediction System", page_icon="⛽", layout="wide")

# Load Trained Model and Data
@st.cache_resource
def load_assets():
    model = joblib.load('brim_gas_model.pkl')
    df = pd.read_csv('brim_gas_sales_data.csv')
    return model, df

model, df = load_assets()

# Sidebar Navigation
st.sidebar.title("Navigation Menu")
app_mode = st.sidebar.selectbox("Choose Section", ["Dashboard Overview", "Make Sales Prediction", "Historical Data Analysis"])

if app_mode == "Dashboard Overview":
    # Enforcing single-line display using HTML with nowrap
    st.markdown(
        """
        <div style="overflow-x: auto; width: 100%;">
            <h1 style='white-space: nowrap; font-size: 26px; margin-bottom: 0px;'>⛽ Brim Gas Nig. Ltd - Sales Prediction System</h1>
            <p style='white-space: nowrap; font-size: 15px; color: #444; font-weight: 600; margin-top: 5px;'>Intelligent Machine Learning Forecasting for LPG Operations in Jalingo, Taraba State</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Recorded Transactions", len(df))
    col2.metric("Average Daily Quantity Sold", f"{df['Quantity_Sold'].mean():.2f} kg")
    col3.metric("Average Unit Price (NGN)", f"₦{df['Unit_Price'].mean():,.2f}")
    
    st.markdown("---")
    st.subheader("Historical Sales Trend (kg)")
    st.line_chart(df.set_index('Date')['Quantity_Sold'])

elif app_mode == "Make Sales Prediction":
    st.title("📊 Future Sales & Revenue Forecasting")
    st.write("Using the current market pricing of **₦4,500 per kg**, adjust the operational parameters below to estimate future demand and revenue generation.")
    
    col1, col2 = st.columns(2)
    with col1:
        unit_price = st.number_input("Unit Price of LPG (NGN/kg)", min_value=3000.0, max_value=8000.0, value=4500.0, step=100.0)
        is_rainy = st.selectbox("Season Type", options=[(1, "Rainy Season (High Demand)"), (0, "Dry Season")], format_func=lambda x: x[1])[0]
    with col2:
        promo = st.selectbox("Promotional Activity Active?", options=[(1, "Yes"), (0, "No")], format_func=lambda x: x[1])[0]
        month = st.slider("Month of Year", min_value=1, max_value=12, value=6)
        day_of_week = st.selectbox("Day of Week", options=[0,1,2,3,4,5,6], format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])

    if st.button("Predict Sales & Revenue", type="primary"):
        input_data = pd.DataFrame([[unit_price, is_rainy, promo, month, day_of_week]], 
                                  columns=['Unit_Price', 'Is_Rainy_Season', 'Promotion_Active', 'Month', 'DayOfWeek'])
        prediction = model.predict(input_data)[0]
        estimated_revenue = prediction * unit_price
        
        st.success(f"### Predicted Sales Volume: {prediction:,.2f} kg")
        st.info(f"### Estimated Revenue Generation: ₦{estimated_revenue:,.2f}")

elif app_mode == "Historical Data Analysis":
    st.title("📈 Historical Database & Metrics")
    st.dataframe(df, use_container_width=True)
