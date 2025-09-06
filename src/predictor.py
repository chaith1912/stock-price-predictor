import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

data=yf.download("AAPL", start="2020-01-01", end="2023-01-01")

print(data.head())

data.to_csv("../data/apple_stock.csv")