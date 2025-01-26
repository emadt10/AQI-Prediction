import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import requests
import requests_cache

# Function to fetch AQI data
def fetch_aqi_data():
    today = datetime.utcnow()
    two_years_ago = today - timedelta(days=2 * 365)
    current_unix_time = int(today.timestamp())
    unix_start = int(two_years_ago.timestamp())

    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat=24.8546842&lon=67.0207055&start={unix_start}&end={current_unix_time}&appid=c78b17200559431652d643ad3e0259a9"
    response = requests.get(url)
    raw = response.json()
    
    # Check if the response contains 'list'
    if 'list' not in raw:
        st.error("Error: Unable to fetch AQI data. Please check the API response.")
        return pd.DataFrame()  # Return an empty DataFrame
    
    aqi_df = pd.json_normalize(raw["list"])
    aqi_df['dt'] = pd.to_datetime(aqi_df['dt'], unit='s')
    aqi_df.set_index('dt', inplace=True)
    aqi_df.index = aqi_df.index.tz_localize(None)
    return aqi_df

# Function to fetch weather data
def fetch_weather_data():
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    start_date = datetime.utcnow() - timedelta(days=2 * 365)
    end_date = datetime.utcnow() - timedelta(days=1)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 24.8546842,
        "longitude": 67.0207055,
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }
    response = requests.get(url, params=params)
    weather_data = response.json()
    
    # Check if the weather data contains 'hourly'
    if 'hourly' not in weather_data:
        st.error("Error: Unable to fetch weather data. Please check the API response.")
        return pd.DataFrame()  # Return an empty DataFrame

    weather_df = pd.DataFrame(weather_data["hourly"])
    weather_df["time"] = pd.to_datetime(weather_df["time"])
    weather_df.set_index("time", inplace=True)
    return weather_df

# Streamlit app
def main():
    st.title("AQI Predictions Dashboard")

    # Load the trained model
    try:
        model = joblib.load("aqi_model.pkl")
        st.success("Model loaded successfully.")
    except FileNotFoundError:
        st.error("Error: Model file 'aqi_model.pkl' not found!")
        return

    # Fetch data dynamically
    aqi_data = fetch_aqi_data()
    weather_data = fetch_weather_data()
    
    # Combine AQI and weather data
    combined_data = aqi_data.join(weather_data, how="inner")

    # Debugging: Check the columns of combined_data
    st.write("Columns in combined_data:", combined_data.columns.tolist())

    # Reset the index to ensure 'dt' is accessible
    combined_data.reset_index(inplace=True)

    # Check the DataFrame after reset
    st.write("DataFrame after reset index:\n", combined_data.head())

    # Rename the 'index' column to 'dt' for clarity
    combined_data.rename(columns={'index': 'dt'}, inplace=True)

    # Filter for the last 3 days of data
    three_days_ago = datetime.utcnow() - timedelta(days=3)
    combined_data = combined_data[combined_data['dt'] >= three_days_ago]

    # Check the filtered DataFrame
    st.write("Filtered DataFrame:\n", combined_data)

    # Set the index back to 'dt' for further processing
    combined_data.set_index('dt', inplace=True)

    # Add additional features
    combined_data["hour"] = combined_data.index.hour
    combined_data["day"] = combined_data.index.day
    combined_data["month"] = combined_data.index.month
    combined_data["day_of_week"] = combined_data.index.dayofweek
    combined_data["season"] = combined_data["month"].apply(lambda x: (x % 12 + 3) // 3)
    combined_data["aqi_change_rate"] = combined_data["main.aqi"].diff()

    # Calculate temperature and humidity change rates
    combined_data["temp_change_rate"] = combined_data["temperature_2m"].diff()
    combined_data["humidity_change_rate"] = combined_data["relative_humidity_2m"].diff()

    # Define features and target
    required_features = [
        "hour", "day", "month", "day_of_week", "season",
        "main.aqi", "temperature_2m", "relative_humidity_2m",
        "aqi_change_rate", "temp_change_rate", "humidity_change_rate"
    ]
    
    # Check for missing features
    missing_features = [feature for feature in required_features if feature not in combined_data.columns]
    if missing_features:
        st.error(f"Missing features: {', '.join(missing_features)}")
        return

    X = combined_data[required_features]
    y = combined_data["main.aqi"].shift(-1).fillna(method="ffill")  # Predict next AQI

    # Handle missing values
    if X.isnull().sum().sum() > 0:
        st.warning("Found missing values in the input data. Handling them with imputation.")
        imputer = SimpleImputer(strategy="mean")
        X = pd.DataFrame(imputer.fit_transform(X), columns=required_features, index=combined_data.index)

    # Make predictions
    y_pred = model.predict(X)

    # Prepare the target variable (actual AQI)
    y_true = y[:-1]  # Remove the last entry for matching lengths
    y_pred = y_pred[:-1]

    # Evaluate performance metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Display metrics
    st.subheader("Model Performance Metrics")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    # Display predicted AQI data
    st.subheader("Predicted AQI for the Next 3 Days")
    predictions = pd.DataFrame({
        "Date & Time": combined_data.index[:-1],
        "True AQI": y_true,
        "Predicted AQI": y_pred
    })
    st.dataframe(predictions)

    # Generate a graph for the predictions
    st.subheader("Prediction Graph for the Recent 3 Days")
    plt.figure(figsize=(10, 5))
    plt.plot(predictions["Date & Time"], predictions["True AQI"], label="True AQI", marker="o", color="g")
    plt.plot(predictions["Date & Time"], predictions["Predicted AQI"], label="Predicted AQI", marker="x", color="b")
    plt.title("AQI Prediction Graph")
    plt.xlabel("Date & Time")
    plt.ylabel("AQI Level")
    plt.legend()
    plt.grid()

    # Display the graph in Streamlit
    st.pyplot(plt)

if __name__ == "__main__":
    main()
