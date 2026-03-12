# Welcome to the BinanceTrader Class! 🚀
# This powerful tool is designed to facilitate algorithmic trading on Binance.
# It handles everything from API connections and data fetching to model predictions
# and trade execution. 

# If you have any questions or encounter any issues, please don't hesitate to contact me-
# -on TG: https://t.me/AlrzA_2003. Thank you!


import ccxt
import pandas as pd
import numpy as np
import schedule
import pytz
from pathlib import Path
from typing import Any, Optional, cast
from threading import Thread
from time import sleep
from datetime import datetime
from Indicators import Indicators
from joblib import load
from tensorflow.keras.models import load_model  # type: ignore[reportMissingImports,reportMissingModuleSource]
from Labels import EndType, get_zig_zag
import Downloads as dls # Optional, Import this package just to avoid rewriting the API_key, Secret, timeframe and trading_pair

class BinanceTrader:
    """
    A class for algorithmic trading on Binance using a deep neural network model.

    Attributes
    ----------
    symbol : str
        The currency pair for algorithmic trading (e.g. "LINK/USDT").
    api_key : str
        The API key obtained from a Binance account.
    secret_key : str
        The secret key obtained from a Binance account.
    model_path : str
        The path to the saved deep neural network model on the local computer or server.
    scaler_path : str
        The path to the saved joblib scaler on the local computer or server.
    bar_length : str
        The timeframe of candles to fetch and trade (default is "15m").
    limit : int
        The number of candles to get per each try (default is 1000).
    leverage : int
        The leverage to trade with (default is 2).
    """
    
    def __init__(self, symbol, api_key, secret_key, testnet, model_path, scaler_path, bar_length = "15m", limit = 1000, leverage = 2, trade_mode = "futures", skip_fetch_currencies = True):
        self.symbol = symbol
        self._api_key = api_key
        self._secret = secret_key
        self._testnet = testnet
        self._trade_mode = trade_mode.lower()
        self._skip_fetch_currencies = skip_fetch_currencies
        self._model = self._load_model_compatible(model_path)
        self._scaler = load(scaler_path)
        self.bar_length = bar_length
        self._limit = limit
        self.leverage = leverage
        self.position = 0
        self.units = 0
        self.data: Optional[pd.DataFrame] = None
        self.n = 0
        self.pnl = None
        self._notional = 1
        self._proba = 0.5  # Raw probability output from model
        self._last_processed_candle = None
        self.make_connection()
        self.fetch_balances()
        self.get_data()

    def _load_model_compatible(self, model_path):
        model_file = Path(model_path)
        try:
            return load_model(str(model_file), compile = False, safe_mode = False)
        except Exception as first_exc:
            try:
                return load_model(str(model_file), compile = False)
            except Exception:
                alt_h5 = model_file.with_suffix(".h5")
                if alt_h5.exists():
                    return load_model(str(alt_h5), compile = False)
                raise RuntimeError(
                    f"Failed to load model from {model_file}. "
                    f"Original error: {first_exc}. "
                    "Re-run Preprocessing.py and re-deploy artifacts."
                ) from first_exc
        
    def __repr__(self):
        time = self.current_time()
        return f"{time} | BinanceTrader Class that uses DNN for predictions!"
    
    def current_time(self): # Fetching current time for print out (timezone = Iran)
        tz_iran = pytz.timezone("America/New_York") # Change the timezone if you want to have print out in your local time.
        current_time = datetime.now(tz_iran)
        dis_time = current_time.strftime("%Y-%m-%d | %H:%M:%S")
        return dis_time

    def _build_exchange_config(self):
        cfg = {'enableRateLimit': True, 'timeout': 30000}
        if self._trade_mode == "futures":
            cfg['options'] = {'defaultType': 'future'}
        return cfg

    def _disable_fetch_currencies_if_needed(self, binance):
        if self._skip_fetch_currencies:
            # Avoid private SAPI currency metadata call, which may fail with HTTP 451 in some cloud regions.
            dynamic_binance = cast(Any, binance)
            options = cast(Any, getattr(dynamic_binance, "options", None))
            if options is None:
                options = {}
                setattr(dynamic_binance, "options", options)
            options["fetchCurrencies"] = False

    def _raise_market_load_error(self, exc):
        message = str(exc)
        if ' 451 ' in message or '451  {' in message:
            raise RuntimeError(
                "Binance rejected this cloud region/IP (HTTP 451) while loading markets. "
                "Keep SKIP_FETCH_CURRENCIES=true and, if the error persists, deploy the job in another region."
            ) from exc
        raise exc

    def _raise_futures_setup_error(self, exc):
        message = str(exc)
        if ('\"code\":-4109' in message) or ("account is inactive" in message.lower()):
            mode = "testnet" if self._testnet else "mainnet"
            raise RuntimeError(
                f"Binance Futures account is inactive on {mode}. Activate Futures for this API key, "
                "or switch to testnet credentials and set testnet=True."
            ) from exc
        if ('\"code\":-2015' in message) or ("invalid api-key" in message.lower()):
            mode = "testnet" if self._testnet else "mainnet"
            raise RuntimeError(
                f"Invalid API key/permissions for Binance Futures on {mode}. "
                "If testnet=True, use Binance Futures TESTNET keys. "
                "If using mainnet keys, set testnet=False and enable Futures + trading permissions."
            ) from exc
        raise exc
    
    def make_connection(self): # Make a connection to Binance.com
        cfg = self._build_exchange_config()
        binance = ccxt.binance(cfg)
        binance.set_sandbox_mode(self._testnet) # For test mode, enable this option
        binance.apiKey = self._api_key
        binance.secret = self._secret
        self._disable_fetch_currencies_if_needed(binance)
        try:
            binance.load_markets()
        except Exception as exc:
            self._raise_market_load_error(exc)
        if self._trade_mode == "spot":
            self.binance = binance
            return
        try:
            binance.set_leverage(self.leverage, symbol = self.symbol)
        except Exception as exc:
            self._raise_futures_setup_error(exc)
        self.binance = binance

    def _strategy_spot(self):
        ex = self.binance
        if self._label == 1:  # Strong long signal
            self._sync_spot_position()
            if self.position == 0:  # Only buy if flat
                if self._amount > 0:
                    order = ex.create_market_order(self.symbol, side = "buy", amount = self._amount)
                    sleep(1)
                    self._sync_spot_position()
                    self.print_status("buy", order)
                else:
                    print("Couldn't Trade! Spot quote balance is too low to BUY")
        elif self._label == -1:  # Strong short signal (sell all if long)
            self._sync_spot_position()
            if self.position == 1 and self.units > 0:
                order = ex.create_market_order(self.symbol, side = "sell", amount = self.units)
                sleep(1)
                self._sync_spot_position()
                self.print_status("sell", order)
        # label == 0: indecision zone (0.4-0.6) → hold current position

    def _retry_futures_entry(self, ex, side):
        flag = True
        order = None
        for item in np.arange(self.leverage, self.leverage - 0.5, -0.02):
            try:
                size = round(self._amount * item, 0)
                order = ex.create_market_order(self.symbol, side = side,
                                             amount = size,
                                             params = {"leverage" : self.leverage})
                flag = False
                break
            except Exception:
                sleep(1)
        return flag, order

    def _enter_futures_long(self, ex):
        if self.position == -1:
            order = ex.create_market_order(self.symbol, side = "buy", amount = self.units)
            self.print_status("close_s", order)
            self.position = 0
            sleep(1)
        if self.position == 0:
            flag, order = self._retry_futures_entry(ex, "buy")
            if flag:
                print("Couldn't Trade! Position = BUY")
            if (not flag) and (order is not None):
                self.print_status("buy", order)
                self.units = abs(round(float(ex.fetch_positions(symbols = [self.symbol])[0]["info"]["positionAmt"]), 2))
                self._real_units = self.units
                self._notional = abs(float(ex.fetch_positions(symbols = [self.symbol])[0]["info"]["notional"]))
                self._tp_units = 1
                self.position = 1

    def _enter_futures_short(self, ex):
        if self.position == 1:
            order = ex.create_market_order(self.symbol, side = "sell", amount = self.units)
            self.print_status("close_b", order)
            self.position = 0
            sleep(1)
        if self.position == 0:
            flag, order = self._retry_futures_entry(ex, "sell")
            if flag:
                print("Couldn't Trade! Position = SELL")
            elif (not flag) and (order is not None):
                self.print_status("sell", order)
                self.units = abs(round(float(ex.fetch_positions(symbols = [self.symbol])[0]["info"]["positionAmt"]), 2))
                self._real_units = self.units
                self._notional = abs(ex.fetch_positions(symbols = [self.symbol])[0]["notional"])
                self._tp_units = 1
                self.position = -1
        
    def fetch_balances(self): # Account Balance
        ex = self.binance
        if self._trade_mode == "spot":
            quote = self.symbol.split("/")[1]
            bal = float(ex.fetch_balance()["free"].get(quote, 0) or 0)
            price = ex.fetch_ticker(symbol = self.symbol)["last"]
            am = (bal * 0.99) / price if price else 0
            am = self._safe_amount_to_precision(am)
            self._amount = am
            self._sync_spot_position()
            return

        bal = ex.fetch_balance()["total"]["USDT"]
        price = ex.fetch_ticker(symbol = self.symbol)["last"]
        am = round(round(bal / price, 2) - round(1 / price, 2), 0)
        self._amount = am

    def _safe_amount_to_precision(self, amount):
        if amount <= 0:
            return 0.0
        try:
            return float(self.binance.amount_to_precision(self.symbol, amount))
        except Exception:
            # Dust balances below market min precision are considered non-tradable.
            return 0.0

    def _sync_spot_position(self):
        ex = self.binance
        base = self.symbol.split("/")[0]
        base_amount = float(ex.fetch_balance()["free"].get(base, 0) or 0)
        self.units = self._safe_amount_to_precision(base_amount)
        self.position = 1 if self.units > 0 else 0
        
    def get_data(self): # fetching the data from Binance using API
        ex = self.binance
        dates = []
        prices = []
        for item in ex.fetch_ohlcv(self.symbol, timeframe = self.bar_length, limit = self._limit):
            item = item[:-1]
            date, *pr = item
            dates.append(pd.to_datetime(date, unit = "ms"))
            prices.append(pr)
        df = pd.DataFrame(index = dates, data = prices, columns = ["Open", "High", "Low", "Close"])
        self.data = df
        
    def prepare_data(self): # Preparing Features and Labels
        if self.data is None:
            raise RuntimeError("Data is not loaded. Call get_data() before prepare_data().")
        current_data = self.data
        # `data` is the working copy of OHLC candles used to build features safely.
        data = current_data.copy()
        ind = Indicators(data)
        ind.all_ind()
        pca = ind.add_pca(1)
        df = ind.data.copy()
        df["Feature_1"] = pca
        df["Labels"] = self.zigzag(source = EndType.HIGH_LOW, pct = 2)
        for i in range(1, 13):
            lag = f"label_lag_{i}"
            df[lag] = df["Labels"].shift(i)
        dropping_cols = ['EMA', 'Open', 'High', 'Close', 'Low']
        df.drop(dropping_cols, axis = 1, inplace = True)
        self.data = df
        
    def zigzag(self, source = EndType.HIGH_LOW, pct = 5): # Creating Labels
        if self.data is None:
            raise RuntimeError("Data is not loaded. Call get_data() before zigzag().")
        current_data = self.data
        df = current_data[["Open", "High", "Low", "Close"]].copy()
        zig_zag_results = get_zig_zag(df.itertuples(), end_type = source, percent_change = pct)
        # Access the Zig Zag results
        trend = []
        for result in zig_zag_results:
            if result.point_type == "H":
                trend.append(0)
            elif result.point_type == "L":
                trend.append(1)
            else:
                trend.append(None)
        s = pd.Series(trend, index = df.index)
        s.ffill(inplace = True)
        return s.shift(-1)
    
    def prepare_model(self): # Preparing the stored model for prediction
        if self.data is None:
            raise RuntimeError("Data is not prepared. Call prepare_data() before prepare_model().")
        current_data = self.data
        data = current_data.copy()
        model = self._model
        scaler = self._scaler
        X = cast(np.ndarray, data.iloc[-1].drop("Labels").to_numpy(dtype = float))
        X = scaler.transform(X.reshape(1, len(data.columns) - 1))
        proba = float(model.predict(X, verbose = None)[0, 0])
        # Apply calibrated thresholds instead of simple rounding (0.5)
        # Only take trades with strong conviction to avoid whipsaw in indecision zone
        buy_threshold = getattr(dls, 'buy_threshold', 0.60)
        sell_threshold = getattr(dls, 'sell_threshold', 0.40)
        if proba >= buy_threshold:
            self._label = 1  # Strong long signal
            self._proba = proba
        elif proba <= sell_threshold:
            self._label = -1  # Strong short signal
            self._proba = proba
        else:
            # No action: stay in current position or wait
            self._label = 0
            self._proba = proba
        
        
    def strategy(self): # Opening LONG/SHORT positions based on predictions
        ex = self.binance
        if self._trade_mode == "spot":
            self._strategy_spot()
            return

        if self._label == 1:  # Strong long signal
            self._enter_futures_long(ex)
            
        elif self._label == -1:  # Strong short signal
            self._enter_futures_short(ex)
        # label == 0: indecision zone → no new entries, hold if already in position

    def _tp_step(self, ex, trigger_pct, units_ratio, next_tp_unit, close_side, going):
        un_pnl = ex.fetch_positions(symbols = [self.symbol])[0]["unrealizedPnl"]
        notional = self._notional
        if ((un_pnl >= (notional * (trigger_pct / self.leverage) / 100)) and (self._tp_units == (next_tp_unit - 1))):
            tp_units = round((self._real_units * units_ratio), 0)
            order = ex.create_market_order(self.symbol, side = close_side,
                                         amount = tp_units)
            self.units = self.units - tp_units
            self._tp_units = next_tp_unit
            self.tp_print_out(going, order)
        
    def tp_position(self):
        if self._trade_mode == "spot":
            return
        ex = self.binance
        if self.position in [-1, 1]:
            close_side = "sell" if self.position == 1 else "buy"
            self._tp_step(ex, trigger_pct = 5, units_ratio = (4 / 10), next_tp_unit = 2, close_side = close_side, going = 5)
            self._tp_step(ex, trigger_pct = 7, units_ratio = (3 / 10), next_tp_unit = 3, close_side = close_side, going = 7)
            self._tp_step(ex, trigger_pct = 10, units_ratio = (2 / 10), next_tp_unit = 4, close_side = close_side, going = 10)
                
    def c_number(self):
        self.n += 1
        print(f"{self.n}", end = " | ", flush = True)
        if self.position == 1:
            c_position = "LONG"
        elif self.position == -1:
            c_position = "SHORT"
        else:
            c_position = "NEUTRAL"
        if (self.n % 5) == 0:
            print("\n")
            print("=" * 50)
            print(f"{self.n}th Hour !", f"Current Position : {c_position}", sep = " ---> ")
            print("=" * 50)
            
    def print_status(self, status, order: Any):
        ex = self.binance
        time = self.current_time()

        if self._trade_mode == "spot":
            amount = order.get("filled") or order.get("amount") or 0
            amount = round(float(amount), 6)
            if "buy" in status:
                print("-" * 100)
                print(f"{time} ---> Buying {self.symbol} for : {amount}")
                print("-" * 100)
            elif "sell" in status:
                print("-" * 100)
                print(f"{time} ---> Selling {self.symbol} for : {amount}")
                print("-" * 100)
            return
        
        if "buy" in status:
            self.order_size_1 = round(float(ex.fetch_positions(symbols = [self.symbol])[0]["info"]["positionAmt"]), 2)
            print("-" * 100)
            print(f"{time} ---> Buying {self.symbol} for : ", self.order_size_1)
            print("-" * 100)
            
        elif "sell" in status:
            self.order_size_2 = round(float(ex.fetch_positions(symbols = [self.symbol])[0]["info"]["positionAmt"]), 2)
            print("-" * 100)
            print(f"{time} ---> Selling {self.symbol} for : ", self.order_size_2)
            print("-" * 100)
            
        elif "close_s" in status:
            trades = cast(Any, ex.fetch_my_trades(symbol = self.symbol))
            last_trade = trades[-1]
            info = last_trade["info"]
            order_size_3 = abs(float(info["qty"]))
            order_size_3 = round(order_size_3, 2)
            pnl = abs(float(info["realizedPnl"]))
            print("-" * 100)
            print(f"{time} ---> Closing SELL Position {self.symbol} | for : ", order_size_3)
            print(f"{time} ---> This Trade's PNL {pnl}")
            print(f"{time} ---> Cum P&L Until Now : {self.pnl}")
            print("-" * 100)
            
        elif "close_b" in status:
            trades = cast(Any, ex.fetch_my_trades(symbol = self.symbol))
            last_trade = trades[-1]
            info = last_trade["info"]
            order_size_3 = abs(float(info["qty"]))
            order_size_3 = round(order_size_3, 2)
            pnl = abs(float(info["realizedPnl"]))
            print("-" * 100)
            print(f"{time} ---> Closing BUY Position {self.symbol} | for : ", order_size_3)
            print(f"{time} ---> This Trade's PNL {pnl}")
            print(f"{time} ---> Cum P&L Until Now : {self.pnl}")
            print("-" * 100)
    
    def tp_print_out(self, going, order: Any):
        time = self.current_time()
        amount = float(order["info"]["origQty"])
        print("-" * 100)
        if going == 2:
            print(f"{time}  |  ORDER 2 PERCENT PLACED !")
            print(f"{time}  |  TAKE PROFIT ORDER AMOUNT IS : {amount}")
            print(f"{time}  |  THE REST AMOUNT IS : {self.units}")
        elif going == 5:
            print(f"{time}  |  ORDER 5 PERCENT PLACED !")
            print(f"{time}  |  TAKE PROFIT ORDER AMOUNT IS : {amount}")
            print(f"{time}  |  THE REST AMOUNT IS : {self.units}")
        elif going == 7:
            print(f"{time}  |  ORDER 7 PERCENT PLACED !")
            print(f"{time}  |  TAKE PROFIT ORDER AMOUNT IS : {amount}")
            print(f"{time}  |  THE REST AMOUNT IS : {self.units}")
        elif going == 10:
            print(f"{time}  |  ORDER 10 PERCENT PLACED !")
            print(f"{time}  |  TAKE PROFIT ORDER AMOUNT IS : {amount}")
            print(f"{time}  |  THE REST AMOUNT IS : {self.units}")
        print("-" * 100)
        
    def all_func(self):
        self.run_single_cycle(only_new_candle = False)

    def _latest_closed_candle(self):
        if self.data is None or len(self.data.index) == 0:
            return None
        if len(self.data.index) >= 2:
            return self.data.index[-2]
        return self.data.index[-1]

    def run_single_cycle(self, only_new_candle = True):
        self.fetch_balances()
        self.get_data()

        latest_closed = self._latest_closed_candle()
        if only_new_candle and latest_closed is not None and latest_closed == self._last_processed_candle:
            print(f"{self.current_time()} | No new closed candle. Skipping cycle.")
            return False

        previous_position = self.position
        self.prepare_data()
        self.prepare_model()
        self.strategy()
        self._last_processed_candle = latest_closed

        print(
            f"{self.current_time()} | Cycle done | label={self._label} | "
            f"position={previous_position}->{self.position}"
        )
        return True
        
    def refreshing_data(self):
        while True:
            tz_utc = pytz.timezone("UTC")
            current_time_utc = datetime.now(tz_utc)
            time = self.current_time()
            if (current_time_utc.minute % 15 == 0) and (current_time_utc.second > 5):
                print(f"{time}  ---->   Start Streaming !!!")
                print("=" * 110, end = "\n\n")
                break
            else:
                sleep(20)
        schedule.every(1).minutes.do(self.fetch_balances)
        schedule.every(15).minutes.do(self.all_func)
#         schedule.every(1).minutes.do(self.tp_position) # Enable it, if you want to have TP positions
        schedule.every(1).hours.do(self.c_number)
        while True:
            schedule.run_pending()
            sleep(10)
            
            
symbol = dls.trading_pair.replace("USDT", "/USDT") # The currency pair you want to trade ---------> "BaseCurrency/QuoteCurrency"
api_key = dls.api_key # The API key you got from Binance
secret = dls.secret # The Secret key you got from Binance
testnet = dls.testnet # Keep runtime mode aligned with Downloads.py
trade_mode = getattr(dls, "trade_mode", "futures") # "spot" or "futures"
skip_fetch_currencies = getattr(dls, "skip_fetch_currencies", True)
single_run = getattr(dls, "single_run", False)
max_cycles_per_run = getattr(dls, "max_cycles_per_run", 1)
sleep_between_cycles_sec = getattr(dls, "sleep_between_cycles_sec", 30)
only_new_candle = getattr(dls, "only_new_candle", True)
model_path = "model.keras" # The path of saved model in your local computer or server
scaler_path = "scaler.joblib" # The path of saved scaler in your local computer or server
bar_length = dls.timeframe # timeframe
limit = 500 # Number of candles you need (based on how many NaN values you have)
leverage = 2 # Desired leverage


trader = BinanceTrader(symbol, api_key, secret, testnet, model_path, scaler_path, bar_length, limit, leverage, trade_mode, skip_fetch_currencies)
print(trader)
if single_run:
    # Cloud scheduler mode: execute a bounded amount of cycles and exit.
    for cycle in range(max_cycles_per_run):
        print(f"{trader.current_time()} | Starting cycle {cycle + 1}/{max_cycles_per_run}")
        trader.run_single_cycle(only_new_candle = only_new_candle)
        if cycle < (max_cycles_per_run - 1):
            sleep(sleep_between_cycles_sec)
    print(f"{trader.current_time()} | Single run completed")
else:
    thread = Thread(target = trader.refreshing_data)
    thread.start()

