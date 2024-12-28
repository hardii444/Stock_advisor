# Stock Trading Advisor

This is a **Stock Trading Advisor** application built using Python and Streamlit. The tool analyzes stock data and provides technical insights to assist in trading decisions. It leverages financial indicators such as Simple Moving Average (SMA), Exponential Moving Average (EMA), Relative Strength Index (RSI), MACD, and Bollinger Bands for its analysis.

## Features

- **Technical Analysis**: 
  - SMA and EMA to identify trends.
  - RSI to determine overbought/oversold conditions.
  - MACD to assess trend strength and reversals.
  - Bollinger Bands for price volatility.
- **Interactive UI**: Enter stock symbols or select from popular stocks to get insights and recommendations.
- **Recommendations**: Based on the analysis, the tool provides actionable suggestions for trading decisions.

## How to Use

1. **Enter Stock Symbol**: Input the stock symbol (e.g., `AAPL` for Apple, `MSFT` for Microsoft).
2. **Analyze**: Click the "Analyze" button to fetch and analyze stock data.
3. **View Results**: Explore the detailed technical analysis, including trends, signals, and a summarized recommendation.

## Requirements

To run the project locally, ensure you have the following installed:

- Python 3.8 or higher
- Streamlit
- pandas
- numpy
- yfinance
- matplotlib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-trading-advisor.git
   cd stock-trading-advisor
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
.
├── app.py               # Main application script
├── README.md            # Documentation file
├── requirements.txt     # Python dependencies
└── utils                # Helper functions and modules
    ├── data_fetcher.py  # Fetches stock data using yfinance
    ├── indicators.py    # Implements technical indicators
    └── recommender.py   # Generates trading recommendations
```

## Contributing

Contributions are welcome! If you'd like to improve the project or fix any issues:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch-name`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io) for providing an easy-to-use web app framework.
- [Yahoo Finance API](https://pypi.org/project/yfinance/) for stock data.

---

### Note
This application is for educational purposes only. It does not constitute financial advice. Use at your own discretion.

---

Happy Trading! 🚀
