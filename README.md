
---

# 🌾 Commodity Price Prediction Web App

This project is a web-based application that predicts commodity prices using machine learning. The app uses **Streamlit** for the frontend and backend integration and allows users to input the district, commodity, and date to predict future prices of agricultural commodities.

## Features

- **Machine Learning Model**: Built using **CatBoost** to predict commodity prices.
- **Interactive Interface**: User-friendly interface built with Streamlit.
- **Custom Input Fields**: Enter district, commodity name, and date for a price prediction.
- **Data Preprocessing**: Automatically preprocesses and cleans input data.
- **Historical Price Trends**: (Optional) Display historical commodity prices.

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: CatBoost for machine learning model
- **Data**: CSV datasets containing historical commodity price data

## Getting Started

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.7+
- Pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/yourusername/commodity-price-prediction.git
cd commodity-price-prediction
```

### Install Dependencies

Install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Run the App

To start the Streamlit app locally, run the following command:

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501` in your web browser.

### Dataset

Place your CSV datasets (containing historical commodity prices) in a folder named `datasets` in the root directory. The app will read the CSV files automatically.

---

## How to Use

1. Open the app in your web browser (`http://localhost:8501`).
2. Enter the **District Name**, **Commodity Name**, and **Date** using the input fields.
3. Click the **Predict** button to see the predicted price.
4. (Optional) View historical price trends in the app.

---

## Project Structure

```plaintext
📁 commodity-price-prediction/
├── 📁 datasets/             # Folder for CSV files containing commodity price data
├── 📁 .streamlit/           # Streamlit configuration files
│   └── config.toml          # Streamlit theme configuration
├── app.py                   # Main Streamlit app
├── requirements.txt         # List of Python dependencies
└── README.md                # Project README
```

---

## Model Overview

The app uses a **CatBoost** model trained on historical commodity price data. The model predicts future prices based on district, commodity, and date, using features like cyclical date encoding.

### Model Training Steps

1. **Data Preprocessing**: Handles missing values and feature engineering for date-related features.
2. **Model Selection**: **CatBoost** is used for handling categorical and numerical data efficiently.
3. **Feature Engineering**: Date features like month, day of the week, and week of the year are encoded.
4. **Training**: The model is trained on historical price data and saved for future predictions.

---

## Sample Prediction

After entering the required inputs (district, commodity, and date), the app will display a prediction like:

```
Predicted price for 'Masur Dal' in 'Prayagraj' on 2024-05-26: Rs. 5300.45 per quintal
```

---

## Deployment

You can deploy the Streamlit app using one of the following methods:

### Deploy on Streamlit Cloud

1. Push your code to a GitHub repository.
2. Go to [Streamlit Cloud](https://share.streamlit.io/).
3. Link your GitHub repository and deploy the app.

### Deploy on Heroku

1. Install the Heroku CLI.
2. Add a `requirements.txt` and a `Procfile` to your project directory.

**Procfile:**

```plaintext
web: streamlit run app.py --server.port $PORT
```

3. Push your code to a Heroku app:

```bash
heroku create
git push heroku master
heroku open
```

---

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- Data Source: Historical commodity price data 

---

## Contributing

Contributions are welcome! If you'd like to contribute, feel free to open a pull request or submit issues.

---

## Acknowledgments

Thanks to the open-source community for tools like **Streamlit** and **CatBoost** that made this project possible.

---

Feel free to update the placeholder links (like `https://github.com/yourusername/commodity-price-prediction.git`) with the actual URLs for your repository. You can also add more content specific to your project as you see fit.

This `README.md` file is designed to be clean, professional, and easy to read in any repository. It should display properly on GitHub and be easy for contributors or users to understand. Let me know if you need any further modifications!
