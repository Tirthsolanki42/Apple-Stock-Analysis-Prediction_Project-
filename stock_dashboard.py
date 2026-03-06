
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("📈 Stock Price Analysis & Prediction Dashboard")
st.sidebar.header("Settings")

# Sidebar inputs
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))

# Load data
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.columns = df.columns.get_level_values(0)
    return df

stock = load_data(ticker, start_date, end_date)

# Feature engineering
stock["Returns"] = stock["Close"].pct_change()
stock["Range"] = stock["High"] - stock["Low"]
stock["MA10"] = stock["Close"].rolling(window=10).mean()
stock["MA50"] = stock["Close"].rolling(window=50).mean()
stock["MA200"] = stock["Close"].rolling(window=200).mean()
stock["Volume_Change"] = stock["Volume"].pct_change()
stock["Tomorrow"] = stock["Close"].shift(-1)
stock["Target"] = (stock["Tomorrow"] > stock["Close"]).astype(int)
stock_clean = stock.dropna()

# Section 1 - Price Chart
st.subheader("📊 Closing Price & Moving Averages")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(stock_clean["Close"], label="Close Price", alpha=0.6)
ax.plot(stock_clean["MA50"], label="50-Day MA", color="orange")
ax.plot(stock_clean["MA200"], label="200-Day MA", color="red")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Section 2 - Key Stats
st.subheader("📋 Key Statistics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${stock_clean['Close'].iloc[-1]:.2f}")
col2.metric("Average Price", f"${stock_clean['Close'].mean():.2f}")
col3.metric("Highest Price", f"${stock_clean['High'].max():.2f}")
col4.metric("Lowest Price", f"${stock_clean['Low'].min():.2f}")

# Section 3 - ML Model
st.subheader("🤖 ML Prediction Model")
features = ["Returns", "Range", "MA10", "MA50", "Volume_Change"]
X = stock_clean[features]
y = stock_clean["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.metric("Model Accuracy", f"{accuracy:.2%}")

# Tomorrow prediction
latest = stock_clean[features].iloc[-1].values.reshape(1, -1)
prediction = model.predict(latest)[0]
if prediction == 1:
    st.success("📈 Model predicts price will GO UP tomorrow")
else:
    st.error("📉 Model predicts price will GO DOWN tomorrow")

# Section 4 - Feature Importance
st.subheader("🔍 Feature Importance")
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
fig2, ax2 = plt.subplots(figsize=(8, 3))
importance.plot(kind="barh", ax=ax2, color="steelblue")
ax2.set_xlabel("Importance Score")
ax2.grid(True)
st.pyplot(fig2)
