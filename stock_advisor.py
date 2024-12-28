import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np

class StockAnalyzer:
    def __init__(self):
        self.base_url = "https://www.alphavantage.co/query"

    def get_stock_data(self, symbol):
        """Get stock data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1y")
            df.columns = [col.lower() for col in df.columns]
            return df
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def calculate_sma(self, prices, period=20):
        """Calculate Simple Moving Average (SMA)"""
        return prices.rolling(window=period).mean()

    def calculate_ema(self, prices, period=20):
        """Calculate Exponential Moving Average (EMA)"""
        return prices.ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index (RSI)"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices):
        """Calculate Moving Average Convergence Divergence (MACD)"""
        ema_12 = self.calculate_ema(prices, period=12)
        ema_26 = self.calculate_ema(prices, period=26)
        macd = ema_12 - ema_26
        signal = self.calculate_ema(macd, period=9)
        return macd, signal

    def calculate_bollinger_bands(self, prices, period=20):
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(prices, period=period)
        rolling_std = prices.rolling(window=period).std()
        upper_band = sma + (rolling_std * 2)
        lower_band = sma - (rolling_std * 2)
        return upper_band, lower_band

    def analyze_stock(self, symbol):
        """Analyze stock and generate recommendation"""
        df = self.get_stock_data(symbol)
        if df is None or len(df) < 50:
            return None, None, None

        # Calculate technical indicators
        df['SMA_20'] = self.calculate_sma(df['close'], period=20)
        df['EMA_20'] = self.calculate_ema(df['close'], period=20)
        df['RSI'] = self.calculate_rsi(df['close'], period=14)
        df['MACD'], df['MACD_signal'] = self.calculate_macd(df['close'])
        df['upper_band'], df['lower_band'] = self.calculate_bollinger_bands(df['close'], period=20)

        latest = df.iloc[-1]  # Latest data point

        # Current Price and Indicator analysis
        current_price = latest['close']
        sma_20 = latest['SMA_20']
        ema_20 = latest['EMA_20']
        rsi = latest['RSI']
        macd = latest['MACD']
        macd_signal = latest['MACD_signal']
        upper_band = latest['upper_band']
        lower_band = latest['lower_band']

        analysis = {
            'Current Price': round(current_price, 2),
            'SMA_20': round(sma_20, 2),
            'EMA_20': round(ema_20, 2),
            'RSI': round(rsi, 2),
            'MACD': round(macd, 2),
            'MACD Signal': round(macd_signal, 2),
            'Upper Band': round(upper_band, 2),
            'Lower Band': round(lower_band, 2),
        }

        # Generate recommendation based on multiple indicators
        recommendation = self.generate_recommendation(sma_20, ema_20, rsi, macd, macd_signal, current_price, upper_band, lower_band)

        return analysis, recommendation, df

    def generate_recommendation(self, sma_20, ema_20, rsi, macd, macd_signal, current_price, upper_band, lower_band):
        """Generate recommendation based on multiple indicators"""
        action = []
        reason = []

        # Check for Bullish or Bearish signals based on each indicator
        if sma_20 > current_price and ema_20 > current_price:
            action.append('Bearish (SMA and EMA indicate downtrend)')
            reason.append('Stock is trading below the 20-day SMA and EMA')
        elif sma_20 < current_price and ema_20 < current_price:
            action.append('Bullish (SMA and EMA indicate uptrend)')
            reason.append('Stock is trading above the 20-day SMA and EMA')
        else:
            action.append('Neutral (Mixed SMA and EMA signals)')
            reason.append('SMA and EMA are in a neutral position')

        # RSI Analysis
        if rsi > 70:
            action.append('Bearish (RSI indicates overbought condition)')
            reason.append('RSI > 70, stock may be overbought')
        elif rsi < 30:
            action.append('Bullish (RSI indicates oversold condition)')
            reason.append('RSI < 30, stock may be oversold')
        else:
            action.append('Neutral (RSI in normal range)')
            reason.append('RSI is between 30 and 70, indicating a normal range')

        # MACD Analysis
        if macd > macd_signal:
            action.append('Bullish (MACD above signal line)')
            reason.append('MACD line is above the signal line, indicating a potential uptrend')
        elif macd < macd_signal:
            action.append('Bearish (MACD below signal line)')
            reason.append('MACD line is below the signal line, indicating a potential downtrend')
        else:
            action.append('Neutral (MACD crossing signal line)')
            reason.append('MACD line is crossing the signal line, indicating indecision')

        # Bollinger Bands Analysis
        if current_price > upper_band:
            action.append('Bearish (Price above upper band)')
            reason.append('Price is above the upper Bollinger Band, suggesting overbought conditions')
        elif current_price < lower_band:
            action.append('Bullish (Price below lower band)')
            reason.append('Price is below the lower Bollinger Band, suggesting oversold conditions')
        else:
            action.append('Neutral (Price within Bollinger Bands)')
            reason.append('Price is within the Bollinger Bands, indicating normal volatility')

        return {'action': ', '.join(action), 'reason': ', '.join(reason)}

def plot_stock_chart(df, symbol):
    """Plot stock data with indicators"""
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC'
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA_20'],
        name='20 Day SMA',
        line=dict(color='orange')
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['EMA_20'],
        name='20 Day EMA',
        line=dict(color='blue')
    ))

    fig.update_layout(
        title=f'{symbol} Stock Analysis',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_white',
        height=600
    )

    return fig

def main():
    st.set_page_config(page_title="Stock Trading Advisor", layout="wide")

    st.title("📈 Stock Trading Advisor")

    # Initialize analyzer
    analyzer = StockAnalyzer()

    # Create two columns for the search options
    col1, col2 = st.columns([2, 3])

    with col1:
        search_method = st.radio("Choose search method:", ["Search Bar", "Popular Stocks"])

        if search_method == "Search Bar":
            symbol = st.text_input(
                "Enter stock symbol:",
                help="""
                Examples:
                - US Stocks: AAPL, MSFT, GOOGL
                - Indian Stocks: RELIANCE.NS, TCS.NS, INFY.NS
                - UK Stocks: VOD.L, BP.L
                - Canadian Stocks: SHOP.TO
                """
            ).upper().strip()
            
            if symbol:
                try:
                    ticker = yf.Ticker(symbol)
                    # Try to get recent data
                    recent_data = ticker.history(period='1d')
                    
                    if recent_data.empty:
                        st.error(f"No data found for symbol '{symbol}'. Please check if the symbol is correct.")
                        symbol = ""
                    else:
                        info = ticker.info
                        company_name = info.get('longName', symbol)
                        st.success(f"Found: {company_name}")
                        # Display some basic info
                        market_price = info.get('regularMarketPrice', 'N/A')
                        currency = info.get('currency', 'N/A')
                        st.info(f"Current Price: {market_price} {currency}")
                except Exception as e:
                    st.error(f"Error: Unable to fetch data for '{symbol}'. Please verify the symbol.")
                    st.info("Make sure to add the correct suffix for non-US stocks (.NS for Indian, .L for London, .TO for Toronto)")
                    symbol = ""

        else:
            popular_stocks = {
                "US Stocks": {
                    "Apple (AAPL)": "AAPL",
                    "Microsoft (MSFT)": "MSFT",
                    "Google (GOOGL)": "GOOGL",
                    "Amazon (AMZN)": "AMZN",
                    "Tesla (TSLA)": "TSLA",
                },
                "Indian Stocks": {
                    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
                    "Tata Consultancy Services (TCS.NS)": "TCS.NS",
                    "Infosys (INFY.NS)": "INFY.NS",
                    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
                    "ITC Limited (ITC.NS)": "ITC.NS"
                }
            }
            
            # First select the market
            market = st.selectbox("Select Market:", options=list(popular_stocks.keys()))
            
            # Then select the stock from that market
            stock_options = popular_stocks[market]
            selected_stock = st.selectbox("Select a stock:", options=list(stock_options.keys()))
            symbol = stock_options[selected_stock]

    if symbol:
        if st.button("Analyze"):
            with st.spinner(f"Analyzing {symbol}..."):
                analysis, recommendation, df = analyzer.analyze_stock(symbol)

                if analysis and recommendation and df is not None:
                    # Get company info
                    try:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        company_name = info.get('longName', symbol)
                        currency = info.get('currency', 'USD')
                        currency_symbol = '₹' if currency == 'INR' else '$'
                    except:
                        company_name = symbol
                        currency_symbol = '$'

                    st.markdown(f"### {company_name} Analysis")
                    rec_col1, rec_col2 = st.columns(2)

                    with rec_col1:
                        st.metric("Action", recommendation['action'])
                    with rec_col2:
                        st.metric("Current Price", f"{currency_symbol}{analysis['Current Price']}")

                    st.plotly_chart(plot_stock_chart(df, company_name), use_container_width=True)

                    st.markdown("### Technical Analysis")
                    analysis_col1, analysis_col2 = st.columns(2)

                    with analysis_col1:
                        st.write(f"SMA (20-day): {currency_symbol}{analysis['SMA_20']}")
                        st.write(f"EMA (20-day): {currency_symbol}{analysis['EMA_20']}")
                        st.write(f"RSI: {analysis['RSI']}")
                        st.write(f"MACD: {analysis['MACD']}")
                    with analysis_col2:
                        st.write(f"MACD Signal: {analysis['MACD Signal']}")
                        st.write(f"Upper Band: {currency_symbol}{analysis['Upper Band']}")
                        st.write(f"Lower Band: {currency_symbol}{analysis['Lower Band']}")

                    st.markdown("### Recommendation Rationale")
                    st.write(recommendation['reason'])

if __name__ == "__main__":
    main()