# commodity_price_prediction_final_fixed_v3.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
from dateutil import parser
import warnings
import logging
import glob
import joblib
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """
    Load the data from CSV files in the 'datasets' folder.
    """
    folder_path = 'datasets'  # Specify the folder where CSV files are stored

    # Get a list of all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        logging.error(f"No CSV files found in the '{folder_path}' folder.")
        return None

    dataframes = []
    for file_path in csv_files:
        logging.info(f"Loading file: {file_path}")
        # Read the CSV file with proper parsing
        df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')

        # Extract commodity name from filename if not present
        if 'Commodity' not in df.columns or df['Commodity'].isnull().all():
            commodity_name = os.path.splitext(os.path.basename(file_path))[0]
            df['Commodity'] = commodity_name

        dataframes.append(df)

    combined_data = pd.concat(dataframes, ignore_index=True)
    logging.info(f"Data loaded. Shape: {combined_data.shape}")
    return combined_data

    """
    Load the data from CSV files in the 'datasets' folder.
    """
    folder_path = 'datasets'  # Specify the folder where CSV files are stored

    # Get a list of all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        logging.error(f"No CSV files found in the '{folder_path}' folder.")
        return None

    dataframes = []
    for file_path in csv_files:
        logging.info(f"Loading file: {file_path}")
        # Read the CSV file with proper parsing
        df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')


        # Extract commodity name from filename if not present
        if 'Commodity' not in df.columns or df['Commodity'].isnull().all():
            commodity_name = os.path.splitext(os.path.basename(file_path))[0]
            df['Commodity'] = commodity_name

        dataframes.append(df)

    combined_data = pd.concat(dataframes, ignore_index=True)
    logging.info(f"Data loaded. Shape: {combined_data.shape}")
    return combined_data

def parse_date(date_str):
    """
    Robust date parsing function to handle various date formats.
    """
    try:
        return pd.to_datetime(date_str, dayfirst=True)
    except:
        try:
            return pd.to_datetime(date_str, dayfirst=False)
        except:
            try:
                return parser.parse(date_str, fuzzy=True)
            except:
                return np.nan

def preprocess_data(df):
    """
    Preprocess the data:
    - Handle date formats
    - Handle missing values
    - Clean categorical variables
    - Feature engineering
    """
    logging.info(f"Initial data shape: {df.shape}")

    # Handle date formats and parse inconsistent dates
    df['Price Date'] = df['Price Date'].apply(parse_date)
    # Drop rows with invalid dates
    df.dropna(subset=['Price Date'], inplace=True)
    logging.info(f"After date conversion and dropping invalid dates: {df.shape}")

    # Sort data by date
    df.sort_values('Price Date', inplace=True)

    # Handle missing price values
    price_cols = ['Min Price(Rs./Quintal)', 'Max Price(Rs./Quintal)', 'Modal Price(Rs./Quintal)']
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Impute missing price values if possible
    df[price_cols] = df[price_cols].fillna(method='ffill').fillna(method='bfill')
    df.dropna(subset=['Modal Price(Rs./Quintal)'], inplace=True)
    logging.info(f"After handling missing price values: {df.shape}")

    # Clean categorical variables (handle typos and inconsistent cases)
    categorical_cols = ['District Name', 'Market Name', 'Variety', 'Grade', 'Commodity']
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Replace missing categorical values
    df[categorical_cols] = df[categorical_cols].fillna('unknown')

    # Feature Engineering: Date features
    df['Year'] = df['Price Date'].dt.year
    df['Month'] = df['Price Date'].dt.month
    df['Day'] = df['Price Date'].dt.day
    df['DayOfWeek'] = df['Price Date'].dt.dayofweek
    df['DayOfYear'] = df['Price Date'].dt.dayofyear
    df['WeekOfYear'] = df['Price Date'].dt.isocalendar().week.astype(int)

    # Cyclical encoding for cyclical features
    cyclical_features = {'Month':12, 'DayOfWeek':7, 'DayOfYear':365, 'WeekOfYear':52}
    for feature, period in cyclical_features.items():
        df[f'{feature}_sin'] = np.sin(2 * np.pi * df[feature]/period)
        df[f'{feature}_cos'] = np.cos(2 * np.pi * df[feature]/period)

    logging.info(f"After feature engineering: {df.shape}")

    # Drop any remaining NaNs in target variable
    df.dropna(subset=['Modal Price(Rs./Quintal)'], inplace=True)
    logging.info(f"After dropping NaNs in target variable: {df.shape}")

    # Check for any remaining NaN values in features
    if df.isnull().values.any():
        logging.warning("Data contains NaN values after preprocessing. Handling NaNs...")
        df.fillna(0, inplace=True)
        logging.info(f"After handling remaining NaNs: {df.shape}")
    else:
        logging.info("No NaN values remain in the data.")

    return df

def train_model(df):
    """
    Train a predictive model using CatBoost Regressor with hyperparameter tuning.
    """
    # Split data into features and target
    X = df.drop(['Price Date', 'Modal Price(Rs./Quintal)', 'Market Name'], axis=1)
    y = df['Modal Price(Rs./Quintal)']

    # Identify categorical features
    categorical_cols = ['District Name', 'Variety', 'Grade', 'Commodity']

    # Ensure categorical columns are strings
    for col in categorical_cols:
        X[col] = X[col].astype(str)

    # Remove columns with zero variance
    cols_to_remove = X.columns[X.nunique() <= 1]
    if len(cols_to_remove) > 0:
        logging.info(f"Removing columns with zero variance: {list(cols_to_remove)}")
        X.drop(columns=cols_to_remove, inplace=True)
        # Adjust categorical feature list after dropping columns
        categorical_cols = [col for col in categorical_cols if col in X.columns]

    # Check for NaN values in X and y
    logging.info(f"\nChecking for NaN values in features and target...")
    logging.info(f"Number of NaN values in X: {X.isnull().sum().sum()}")
    logging.info(f"Number of NaN values in y: {y.isnull().sum()}")

    # Ensure there are no NaN values
    if X.isnull().any().any():
        logging.warning("Features contain NaN values. Handling NaNs in features...")
        X.fillna(0, inplace=True)
    if y.isnull().any():
        logging.warning("Target contains NaN values. Handling NaNs in target...")
        y.fillna(y.mean(), inplace=True)

    # Convert all numerical columns to float64
    for col in X.columns:
        if col not in categorical_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # Split into train and test sets chronologically
    split_date = df['Price Date'].quantile(0.8)
    train_indices = df['Price Date'] <= split_date
    test_indices = df['Price Date'] > split_date

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    logging.info(f"Training data shape: {X_train.shape}")
    logging.info(f"Testing data shape: {X_test.shape}")

    # Initialize CatBoost Regressor
    model = CatBoostRegressor(
        eval_metric='MAE',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50
    )

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'iterations': [500],
        'learning_rate': [0.03],
        'depth': [6],
        'l2_leaf_reg': [3]
    }

    # TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=3)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )

    # Fit the model
    grid_search.fit(
        X_train, y_train,
        cat_features=categorical_cols
    )

    # Best parameters
    logging.info(f"Best parameters found: {grid_search.best_params_}")

    # Best model
    best_model = grid_search.best_estimator_

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"\nTest MAE: {mae:.2f}")
    logging.info(f"Test RMSE: {rmse:.2f}")
    logging.info(f"Test R^2 Score: {r2:.2f}")

    # Save feature names for later use
    feature_names = X_train.columns.tolist()
    joblib.dump(feature_names, 'feature_names.pkl')
    # Save categorical features used during training
    joblib.dump(categorical_cols, 'categorical_features.pkl')

    return best_model

def get_cyclical_features(date):
    """
    Create cyclical features for the input date to match the feature engineering.
    """
    year = date.year
    month = date.month
    day_of_week = date.weekday()
    day_of_year = date.timetuple().tm_yday
    week_of_year = date.isocalendar()[1]

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

def predict_price(model):
    """
    Predict price based on user input for district, commodity, and date.
    """
    district = input("Enter the district name: ").strip().lower()
    commodity = input("Enter the commodity name: ").strip().lower()
    input_date_str = input("Enter the date (YYYY-MM-DD): ").strip()

    # Parse input date
    try:
        input_date = datetime.strptime(input_date_str, '%Y-%m-%d')
    except ValueError:
        logging.error("Invalid date format. Please enter the date in YYYY-MM-DD format.")
        return

    # Create a DataFrame for the prediction
    input_data = {
        'District Name': [district],
        'Commodity': [commodity],
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

    # Load feature names and ensure input data has all features
    feature_names = joblib.load('feature_names.pkl')
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0  # Default value for missing features

    # Reorder columns to match the model's expected order
    input_df = input_df[feature_names]

    # Load the list of categorical features used during training
    categorical_features = joblib.load('categorical_features.pkl')

    # Ensure the input data's categorical columns are strings
    for col in categorical_features:
        input_df[col] = input_df[col].astype(str)

    # Create a CatBoost Pool object, specifying categorical features
    pool = Pool(input_df, cat_features=categorical_features)

    # Make prediction
    predicted_price = model.predict(pool)

    # Show the predicted price
    print(f"Predicted price for {commodity} in {district} on {input_date_str}: Rs. {predicted_price[0]:.2f} per quintal")

def main():
    # Check if all necessary files exist
    model_filename = 'commodity_price_model.cbm'
    feature_names_file = 'feature_names.pkl'
    categorical_features_file = 'categorical_features.pkl'

    if os.path.exists(model_filename) and os.path.exists(feature_names_file) and os.path.exists(categorical_features_file):
        # Load the model
        logging.info("Loading the trained model and necessary files...")
        model = CatBoostRegressor()
        model.load_model(model_filename)
        logging.info("Model and necessary files loaded successfully.")
    else:
        # Load data
        logging.info("Loading data...")
        data = load_data()
        if data is None:
            logging.error("Data loading failed. Exiting...")
            return
        logging.info("Data loaded successfully.")

        # Preprocess data
        logging.info("\nPreprocessing data...")
        data = preprocess_data(data)
        logging.info("Data preprocessed successfully.")

        # Check if data is empty after preprocessing
        if data.empty:
            logging.error("No data available after preprocessing. Please check your input files and preprocessing steps.")
            return

        # Train model
        logging.info("\nTraining model...")
        model = train_model(data)
        logging.info("Model trained successfully.")

        # Save the model
        model.save_model(model_filename)
        logging.info(f"Model saved as '{model_filename}'.")

    # Ask for district, commodity, and date to predict price
    predict_price(model)

if __name__ == "__main__":
    main()
