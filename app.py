
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit_authenticator as stauth
from datetime import datetime

st.set_page_config(page_title="AI Trading Bot", layout="wide")

names = ["Demo User"]
usernames = ["demo"]

hashed_passwords = {
    "demo": "$2b$12$KIXuPMQXUQmYpjpMF5FiUOlbgLTy1MsoBv9v7pyEakzgy/q5AUWYi"
}

authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "trading_dashboard", "abcdef", cookie_expiry_days=1)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status:
    authenticator.logout("Logout", "sidebar")
    st.title("📈 AI-Powered Trading Bot Simulator")

    user_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", value="AAPL")

    try:
        data = yf.download(tickers=user_ticker, period="1d", interval="1m")
        if data.empty or "Close" not in data.columns:
            raise ValueError("No valid price data retrieved. Please check your internet connection or ticker symbol.")
    except Exception as e:
        st.warning(f"Data retrieval error: {e}")
        st.info("Loading fallback test data...")
        test_prices = np.linspace(100, 200, 120) + np.random.randn(120) * 5
        data = pd.DataFrame({"Close": test_prices})

    if data is not None:
        prices = data["Close"].to_numpy().flatten()
        n_days = len(prices)

        ACTIONS = ["BUY", "SELL", "HOLD"]
        action_to_index = {a: i for i, a in enumerate(ACTIONS)}

        def compute_indicators(prices):
            sma10 = pd.Series(prices).rolling(window=10).mean().to_numpy()
            delta = np.diff(prices, prepend=prices[0])
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain).rolling(window=14).mean().to_numpy()
            avg_loss = pd.Series(loss).rolling(window=14).mean().to_numpy()
            rs = avg_gain / (avg_loss + 1e-6)
            rsi = 100 - (100 / (1 + rs))
            return sma10, rsi

        sma10, rsi = compute_indicators(prices)

        def get_features(day):
            return np.array([
                prices[day],
                prices[day + 1],
                sma10[day],
                sma10[day + 1],
                rsi[day]
            ])

        def train_bot(episodes=100):
            rewards_over_time = []
            final_cash = 0
            final_shares = 0
            action_counts = {a: 0 for a in ACTIONS}
            trades = []

            for episode in range(episodes):
                cash = 10000
                shares = 0
                total_rewards = 0

                for day in range(1, n_days - 2):
                    state = get_features(day)
                    price_today = prices[day]
                    price_tomorrow = prices[day + 1]

                    action = np.random.choice(ACTIONS)
                    action_counts[action] += 1

                    if action == "BUY" and cash >= price_today:
                        quantity = cash // price_today
                        trades.append((day, "BUY", quantity, price_today))
                        shares = quantity
                        cash -= shares * price_today
                    elif action == "SELL" and shares > 0:
                        trades.append((day, "SELL", shares, price_today))
                        cash += shares * price_today
                        shares = 0

                    portfolio_now = cash + shares * price_today
                    portfolio_future = cash + shares * price_tomorrow
                    reward = portfolio_future - portfolio_now
                    total_rewards += reward

                final_cash = cash
                final_shares = shares
                rewards_over_time.append(total_rewards)

            return rewards_over_time, final_cash, final_shares, action_counts, trades

        episodes = st.slider("Training Episodes", min_value=10, max_value=500, value=100, step=10)
        rewards, cash, shares, action_counts, trades = train_bot(episodes=episodes)

        final_price = float(prices[-1])
        final_value = cash + shares * final_price
        st.success(f"Final Portfolio Value: ${final_value:.2f}")
        st.info(f"Total Reward (Profit/Loss): ${rewards[-1]:.2f}")

        st.subheader("📊 Action Distribution")
        st.bar_chart(pd.Series(action_counts))

        st.subheader("📈 Rewards Over Episodes")
        st.line_chart(rewards)

        st.subheader("🧾 Trade History")
        trade_df = pd.DataFrame(trades, columns=["Day", "Action", "Quantity", "Price"])
        st.dataframe(trade_df)

else:
    if authentication_status is False:
        st.error("Username/password is incorrect")
    elif authentication_status is None:
        st.warning("Please enter your username and password")
