# Hello there! 😊 
# If you're looking to gather data to train your model, this code will help you do just that.
# This script fetches data from Binance Futures, whether you're using the mainnet or the testnet.
# Before running the code, make sure to install the necessary packages with the following command:
# pip install python-binance pandas numpy

# If you encounter any issues or have questions, feel free to reach out to me on Telegram: https://t.me/AlrzA_2003


from binance.client import Client
import pandas as pd
import os
from pathlib import Path


def _load_env_file(env_path: str = ".env"):
    path = Path(env_path)
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _to_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: str, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


_load_env_file()

api_key = os.getenv("API_KEY")
secret = os.getenv("SECRET")

if not api_key or not secret:
    raise RuntimeError(
        "Missing API credentials. Set API_KEY and SECRET in environment variables or .env file."
    )

timeframe = os.getenv("TIMEFRAME", "15m") # Change the timeframe here.
trading_pair = os.getenv("TRADING_PAIR", "BTCUSDT") # Choose your pair here.
testnet = _to_bool(os.getenv("TESTNET", "false"), default = False) # Keep this in sync with BinanceTrader runtime mode.
trade_mode = os.getenv("TRADE_MODE", "spot") # "spot" or "futures"
skip_fetch_currencies = _to_bool(os.getenv("SKIP_FETCH_CURRENCIES", "true"), default = True)
single_run = _to_bool(os.getenv("SINGLE_RUN", "false"), default = False)
max_cycles_per_run = max(1, _to_int(os.getenv("MAX_CYCLES_PER_RUN", "1"), default = 1))
sleep_between_cycles_sec = max(1, _to_int(os.getenv("SLEEP_BETWEEN_CYCLES_SEC", "30"), default = 30))
only_new_candle = _to_bool(os.getenv("ONLY_NEW_CANDLE", "true"), default = True)
# Replace "BTCUSDT" and "15m" with any currency pair and timeframe you want.

if __name__ == "__main__":
    client = Client(api_key = api_key, api_secret = secret, tld = "com", testnet = testnet) 
            # Set testnet to "True" if you're currently working with Binance Future Testnet.
    
    start = str(pd.to_datetime(client._get_earliest_valid_timestamp(trading_pair, timeframe), unit = "ms")) 
            
    
    
    bars = client.futures_historical_klines(symbol = trading_pair, interval = timeframe,
                                            start_str = start, end_str = None, limit = 1000)
    df = pd.DataFrame(bars)
    df["Date"] = pd.to_datetime(df.iloc[:,0], unit = "ms")
    df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                    "Clos Time", "Quote Asset Volume", "Number of Trades",
                    "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
    df = df[["Date", "Open", "High", "Low", "Close"]].copy()
    df.set_index("Date", inplace = True)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors = "coerce")
    df.to_csv("{}_{}.csv".format(trading_pair, timeframe))
    print("Done!")

