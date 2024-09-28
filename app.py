import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from catboost import CatBoostRegressor, Pool
import os
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data function
def load_data():
    folder_path = 'datasets'
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        st.error(f"No CSV files found in the '{folder_path}' folder.")
        return None

    dataframes = []
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        commodity_name = os.path.splitext(os.path.basename(file_path))[0]
        df['Commodity'] = commodity_name
        dataframes.append(df)

    combined_data = pd.concat(dataframes, ignore_index=True)
    return combined_data

# Preprocess the data
def preprocess_data(df):
    df['Price Date'] = pd.to_datetime(df['Price Date'], errors='coerce')
    df.dropna(subset=['Price Date'], inplace=True)
    df.sort_values('Price Date', inplace=True)
    
    price_cols = ['Min Price(Rs./Quintal)', 'Max Price(Rs./Quintal)', 'Modal Price(Rs./Quintal)']
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df[price_cols] = df[price_cols].fillna(method='ffill').fillna(method='bfill')
    df.dropna(subset=['Modal Price(Rs./Quintal)'], inplace=True)

    categorical_cols = ['District Name', 'Market Name', 'Variety', 'Grade', 'Commodity']
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
    
    df[categorical_cols] = df[categorical_cols].fillna('unknown')

    df['Year'] = df['Price Date'].dt.year
    df['Month'] = df['Price Date'].dt.month
    df['Day'] = df['Price Date'].dt.day
    df['DayOfWeek'] = df['Price Date'].dt.dayofweek
    df['DayOfYear'] = df['Price Date'].dt.dayofyear
    df['WeekOfYear'] = df['Price Date'].dt.isocalendar().week.astype(int)

    cyclical_features = {'Month':12, 'DayOfWeek':7, 'DayOfYear':365, 'WeekOfYear':52}
    for feature, period in cyclical_features.items():
        df[f'{feature}_sin'] = np.sin(2 * np.pi * df[feature]/period)
        df[f'{feature}_cos'] = np.cos(2 * np.pi * df[feature]/period)

    return df

@st.cache_resource
def load_model(model_path):
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model

# Get cyclical features
def get_cyclical_features(input_date):
    month = input_date.month
    day_of_week = input_date.weekday()
    day_of_year = input_date.timetuple().tm_yday
    week_of_year = input_date.isocalendar()[1]

    cyclical_features = {
        'Month_sin': np.sin(2 * np.pi * month / 12),
        'Month_cos': np.cos(2 * np.pi * month / 12),
        'DayOfWeek_sin': np.sin(2 * np.pi * day_of_week / 7),
        'DayOfWeek_cos': np.cos(2 * np.pi * day_of_week / 7),
        'DayOfYear_sin': np.sin(2 * np.pi * day_of_year / 365),
        'DayOfYear_cos': np.cos(2 * np.pi * day_of_year / 365),
        'WeekOfYear_sin': np.sin(2 * np.pi * week_of_year / 52),
        'WeekOfYear_cos': np.cos(2 * np.pi * week_of_year / 52)
    }

    return cyclical_features

# Predict the price
def predict_price(model, district, commodity, input_date):
    input_data = {
        'District Name': [district.strip().lower()],
        'Commodity': [commodity.strip().lower()],
        'Variety': ['unknown'],
        'Grade': ['unknown'],
        'Year': [input_date.year],
        'Month': [input_date.month],
        'Day': [input_date.day],
        'DayOfWeek': [input_date.weekday()],
        'DayOfYear': [input_date.timetuple().tm_yday],
        'WeekOfYear': [input_date.isocalendar()[1]],
    }

    # Add cyclical features for the date
    cyclical_features = get_cyclical_features(input_date)
    input_data.update(cyclical_features)

    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)

    # Load feature names and categorical features
    feature_names = joblib.load('feature_names.pkl')
    categorical_features = joblib.load('categorical_features.pkl')

    # Add missing features
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0

    input_df = input_df[feature_names]

    # Create a Pool object
    pool = Pool(input_df, cat_features=categorical_features)

    # Predict price
    predicted_price = model.predict(pool)

    return predicted_price[0]

# Main function
def main():
    st.title("Commodity Price Prediction 🌾💰")
    st.write("Predict future prices of commodities based on district and date.")

    # Input fields for district, commodity, and date
    district = st.text_input("Enter the district name")
    commodity = st.text_input("Enter the commodity name")
    input_date = st.date_input("Select a date")

    model = load_model("commodity_price_model.cbm")

    # Prediction button
    if st.button("Predict"):
        if not district or not commodity:
            st.error("Please fill in all the fields.")
        else:
            predicted_price = predict_price(model, district, commodity, input_date)
            st.success(f"Predicted price for {commodity} in {district} on {input_date}: Rs. {predicted_price:.2f} per quintal")

if __name__ == "__main__":
    main()
