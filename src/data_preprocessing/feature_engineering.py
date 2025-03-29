import pandas as pd
import yfinance as yf
import os


"""
# Feature Engineering Pipeline for Customer Churn Classification

This script enhances the dataset by generating new features related to date differences, stock market trends, and statistical metrics. 

### Key Steps:
1. Date-Based Feature Engineering (`create_date_features`):
   - Computes `customer_time_period` (days between transaction and issuing date).
   - Extracts transaction weekday and quarter.
   - Drops unnecessary columns (`customer_id`, `issuing_date`).

2. Stock Market Data Fetching (`fetch_stock_data`):
   - Retrieves NIFTY 50 historical data using Yahoo Finance API.
   - Collects stock closing prices and computes a 3-month rolling trend.
   - Cleans and formats stock market data for integration.

3. Merging Stock Data (`merge_stock_data`):
   - Merges stock trends with the main dataset based on transaction date.
   - Uses forward-fill to handle missing stock values.

4. Stock-Based Feature Engineering (`create_stock_features`):
   - Calculates `stock_return`: Percentage change in the 3-month trend.
   - Computes `stock_volatility`: Rolling 7-period standard deviation.
   - Imputes missing values with mean values.

5. Complete Feature Engineering Pipeline (`engineer_features`):
   - Combines date-based and stock market features.
   - Loads previously processed data (`preprocess_part1.csv`).
   - Merges newly engineered features with existing preprocessed data.
   - Cleans redundant or missing values before finalizing the dataset.

6. Execution (`__main__` block):
   - Loads the preprocessed dataset (`preprocess_part2.csv`).
   - Converts relevant columns to datetime format.
   - Runs feature engineering and saves the final dataset as `final_dataset.csv`.

"""

def create_date_features(df):
    """
    Generate date-dependent features:
    - customer_time_period: Days between transaction date and issuing date.
    """
    df["customer_time_period"] = (df["date"] - df["issuing_date"]).dt.days
    df["transaction_weekday"] = df["date"].dt.weekday
    df["quarter"] = df["date"].dt.quarter

    # Drop unnecessary columns
    df.drop(columns=["customer_id", "issuing_date"], inplace=True)

    return df

def fetch_stock_data(df):
    """
    Fetch stock market data for NIFTY 50 using Yahoo Finance API.
    - Adjusts the date range to ensure we have relevant past data.
    """
    # Determine min and max dates from the dataset
    min_date = df["date"].min() - pd.DateOffset(months=2)  # Add buffer for historical trends
    max_date = df["date"].max() + pd.DateOffset(months=1)  # Add buffer for forecasting trends

    # Download NIFTY 50 historical data
    stock_data = yf.download("^NSEI", start=min_date, end=max_date, interval="1mo")

    # Keep only closing prices and rename the column
    stock_data = stock_data[['Close']].rename(columns={'Close': 'nifty_50_close'})

    # Convert index to date format and reset index
    stock_data.index = stock_data.index.date
    stock_data = stock_data.reset_index().rename(columns={'index': 'date'})

    # Compute 3-month rolling average trend
    stock_data['nifty_50_trend_3mo'] = stock_data['nifty_50_close'].rolling(3).mean()

    # Drop original close price column
    stock_data.drop(columns=["nifty_50_close"], inplace=True)
    
    stock_data.columns = stock_data.columns.droplevel(1)  # Removes 'Ticker' level
    stock_data = stock_data.reset_index()

    return stock_data

def merge_stock_data(df, stock_data):
    """
    Merge stock data with the transaction dataset:
    - Ensures date formats match before merging.
    - Uses forward-fill to handle missing values.
    """
    # Convert transaction dates to match stock data format
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Merge stock market trends with main dataset
    df = df.merge(stock_data, on="date", how="left")

    # Fill missing values using forward-fill method
    df['nifty_50_trend_3mo'].fillna(method='ffill', inplace=True)
    
    df.drop(columns=["index", "date"], inplace=True)

    return df

def create_stock_features(df):
    """
    Generate stock market-related features:
    - stock_return: Percentage change in the 3-month trend.
    - stock_volatility: Rolling 7-period standard deviation.
    """
    df['stock_return'] = df['nifty_50_trend_3mo'].pct_change()
    df['stock_volatility'] = df['nifty_50_trend_3mo'].rolling(7).std()

    # Drop the 3-month trend column after feature extraction
    df.drop(columns=["nifty_50_trend_3mo"], inplace=True)
    
    df.fillna({
    'stock_return': df['stock_return'].mean(),  
    'stock_volatility': df['stock_volatility'].mean()
    }, inplace=True)

    return df

def engineer_features(df):
    """
    Complete feature engineering pipeline:
    - Generate date-related features.
    - Fetch and merge stock market data.
    - Generate stock-based statistical features.
    """
    df = create_date_features(df)
    stock_data = fetch_stock_data(df)
    df = merge_stock_data(df, stock_data)
    
    file_path = os.path.join(os.getcwd().split("customer-churn-classification")[0], 'customer-churn-classification/data/processed/preprocess_part1.csv')
    df_part = pd.read_csv(file_path)
    
    df_combined = pd.concat([df_part, df], axis=1)
    
    
    df = create_stock_features(df_combined)

    return df

if __name__ == "__main__":
    # Load processed dataset (assuming preprocessing has been done)
    file_path = os.path.join(os.getcwd().split("customer-churn-classification")[0], 'customer-churn-classification/data/processed/preprocess_part2.csv')
    df = pd.read_csv(file_path)

    # Convert date columns to datetime
    df["date"] = pd.to_datetime(df["date"])
    df["issuing_date"] = pd.to_datetime(df["issuing_date"])

    # Perform feature engineering
    df_transformed = engineer_features(df)
    
    df_transformed.drop(columns=[col for col in df_transformed.columns if "Unnamed" in col], inplace=True)

    # Save the transformed dataset
    df_transformed.to_csv(os.path.join(os.getcwd().split("customer-churn-classification")[0], 'customer-churn-classification/data/processed/final_dataset.csv'), index=False)
    print("Feature engineering completed successfully!")
