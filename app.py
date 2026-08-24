import torch
import pandas as pd

from collections import deque
from datetime import datetime, time
from zoneinfo import ZoneInfo

from data_filler import get_unix_timestamps
from model_query import setup_model, query_model, print_prediction
from ibkr_client import IBKRClient
from setup import API_KEY, BAR_WIDTH, INPUT_FEATURES, TARGET_FEATURES, DEBUG
from setup import path_stockGPT_B1r, path_stockGPT_B5r
from setup import StockGPT_cfg

class App():
    def __init__(self):
        self.api_key = API_KEY

        self.tz = ZoneInfo("America/New_York")
        self.date = datetime.now(self.tz).date()
        self.market_open_s = int(datetime.combine(self.date, time(9, 30),
                                            tzinfo=self.tz).timestamp())
        self.market_close_s = int(datetime.combine(self.date, time(16, 0),
                                          tzinfo=self.tz).timestamp())
        self.selected_tickers = set()
        self.input_features = INPUT_FEATURES
        self.target_features = TARGET_FEATURES

        #* MODELS -----------------------------
        self.stockGPT_B1, self.B1_device = setup_model(path_stockGPT_B1r)
        self.stockGPT_B5, self.B5_device = setup_model(path_stockGPT_B5r)

        #* IBKRClient -------------------------
        self.client = IBKRClient(self)

        #* DATA ------------------------------
        self.data = AppData(self)

    def get_apikey(self):
        return self.api_key
    
    def get_date(self):
        return self.date

    def get_inputfeatures(self):
        return self.input_features

    def get_targetfeatures(self):
        return self.target_features

    def select_ticker(self, ticker: str):
        if DEBUG:
            print(f"Selecting {ticker}")
        self.client.add_ticker(ticker)

    def unselect_ticker(self, ticker: str):
        base_id = self.client.remove_ticker(ticker)
        if base_id != -1:
            self.data.remove_ticker(base_id)

    def show_selected_tickers(self):
        print(self.client.tickers)

    def query_model(self, df, bar_width: int):
        model = self.stockGPT_B1 if bar_width == 1 else self.stockGPT_B5
        tensor = torch.tensor(
            df[self.input_features].to_numpy(),
            dtype=torch.float32
        ).unsqueeze(0)
        prev, residual, std = query_model(model, tensor)
        return prev, residual, std

    def get_predictions(self, ticker):
        base_id = self.client.ticker_ids[ticker]
        return [self.data.id_pred_bars[base_id], self.data.id_pred_bars[base_id+1]]

    def print_predictions(self, preds: list):
        for pred in preds:
            print_prediction(pred["prev"], pred["res"], pred["std"], self.input_features, self.target_features)

    def predict_ticker(self, ticker):
        if ticker not in self.client.tickers:
            print("Ticker has not been added yet!")
        else:
            self.print_predictions(self.get_predictions(ticker))

    def execute(self):
        try:
            print("--- StockGPT App-V1 ------------")
            while True:
                command = input(
                    "\n"
                    "1: Add ticker\n"
                    "2: Remove ticker\n"
                    "3: Show tickers\n"
                    "4: Predict ticker\n"
                    "q: Quit\n"
                    "\n"
                    "   >> "
                )
                if command == "1":
                    ticker = input("\nTicker: ").upper()
                    self.select_ticker(ticker)
                elif command == "2":
                    ticker = input("\nTicker: ").upper()
                    self.unselect_ticker(ticker)
                elif command == "3":
                    self.show_selected_tickers()
                elif command == "4":
                    ticker = input("\nTicker: ").upper()
                    self.predict_ticker(ticker)
                elif command.lower() == "q":
                    break
        finally:
            self.client.close()
    
class AppData():
    def __init__(self, app):
        self.app = app
        self.client = app.client
        self.id_eth_bars = {}               #* Extended hours (MyBars)
        self.id_rth_bars = {}               #* Today's Regular Trading hours (MyBars)
        self.id_current_bar = {}            #* Unfinished (MyBar)
        self.id_rolling_features = {}       #* Stores all data for calculating rolled features per id
        self.id_rth_processed_bars = {}     #* --> gets sent to model query as tensor
        self.id_pred_bars = {}              #* Stores predicted bars every update per id
        self.bar_width = {                      #! HARDCODED
            0: 1,
            1: 5
        }
        if DEBUG:
            print("DataState initialized")

    def ibkrbar_to_mybar(self, bar):
        return {
            "t": int(bar.date),
            "vw": float(bar.wap),
            "o": float(bar.open),
            "h": float(bar.high),
            "l": float(bar.low),
            "c": float(bar.close),
            "v": float(bar.volume),
            "n": int(bar.barCount),
        }

    def sort_bar(self, req_id, mybar):
        """
        Sorts bars into rth/eth
        """
        if self.app.market_open_s <= mybar["t"] < self.app.market_close_s:
            self.id_rth_bars[req_id].append(mybar)
            return True
        else:
            self.id_eth_bars[req_id].append(mybar)
            return False

    def initialize_id_history(self, req_id, bar):
        mybar = self.ibkrbar_to_mybar(bar)
        if req_id not in self.id_eth_bars:
            self.id_eth_bars[req_id] = []
            self.id_rth_bars[req_id] = []
            self.id_pred_bars[req_id] = {
                "prev": None,
                "res": None,
                "std": None
            }
        self.sort_bar(req_id, mybar)

    def initialize_id_rolling_features(self, req_id, df):
        """
        Calculates across all eth and rth bars
        """
        if req_id not in self.id_rolling_features:
            self.id_rolling_features[req_id] = []

        ema9 = df["c"].ewm(span=9, adjust=False).mean()
        ema20 = df["c"].ewm(span=20, adjust=False).mean()
        ema12 = df["c"].ewm(span=12, adjust=False).mean()
        ema26 = df["c"].ewm(span=26, adjust=False).mean()
        volumes = deque(df["v"].tail(90).tolist(), maxlen=90)

        self.id_rolling_features[req_id] = {
            "ema9": ema9.iloc[-1],
            "ema12": ema12.iloc[-1],
            "ema20": ema20.iloc[-1],
            "ema26": ema26.iloc[-1],
            "volumes": volumes,
            "volume_sum": sum(volumes),
        }

        if DEBUG:
            print(f"Rolling: {self.id_rolling_features[req_id]}")

    def finish_id_initialization(self, req_id):
        """
        Completes fetching of all recent bars
        """
        rth = self.id_rth_bars[req_id]
        eth = self.id_eth_bars[req_id]

        #* removes last bar if incomplete
        if rth:
            self.id_current_bar[req_id] = rth.pop()
        else:
            self.id_current_bar[req_id] = eth.pop()

        processed_bars_df = DataProcessor.process_data(
            rth + eth,
            self.bar_width[req_id % 2]
        )
        self.initialize_id_rolling_features(req_id, processed_bars_df)
        self.id_rth_processed_bars[req_id] = processed_bars_df[
            (processed_bars_df["t"] >= self.app.market_open_s) &
            (processed_bars_df["t"] < self.app.market_close_s)
        ].copy()

        if DEBUG:
            print(f"eth: {len(eth)}")
            print(f"rth: {len(rth)}")
            print(self.id_rth_processed_bars[req_id])

    def process_bar(self, req_id, mybar, replace_rolling=False):
        """
        Processes a bar ready for model input (in-place) and optionally replaces data in id_rolling_features
        """
        state = self.id_rolling_features[req_id]

        c = mybar["c"]
        v = mybar["v"]

        ema9 = (2 / (9 + 1)) * c + (1 - 2 / (9 + 1)) * state["ema9"]
        ema20 = (2 / (20 + 1)) * c + (1 - 2 / (20 + 1)) * state["ema20"]
        ema12 = (2 / (12 + 1)) * c + (1 - 2 / (12 + 1)) * state["ema12"]
        ema26 = (2 / (26 + 1)) * c + (1 - 2 / (26 + 1)) * state["ema26"]

        volumes = deque(state["volumes"], maxlen=90)
        volume_sum = state["volume_sum"]

        if len(volumes) == 90:
            volume_sum -= volumes[0]

        volumes.append(v)
        volume_sum += v

        av = volume_sum / len(volumes)
        rv = v / av if av != 0 else 0.0

        mybar["ema9"] = ema9
        mybar["ema20"] = ema20
        mybar["macd"] = ema12 - ema26
        mybar["rv"] = rv
        mybar["f"] = 0
        mybar["bar"] = ((mybar["t"] - self.app.market_open_s) // (self.bar_width[req_id % 2] * 60)) + 1

        if replace_rolling:
            self.id_rolling_features[req_id] = {
                "ema9": ema9,
                "ema12": ema12,
                "ema20": ema20,
                "ema26": ema26,
                "volumes": volumes,
                "volume_sum": volume_sum,
            }

    def update_id_data(self, req_id, bar):
        """
        Any new bars after subscription get processed and placed into eth/rth
        """
        if req_id not in self.id_current_bar:
            return #* Prevents updating already unselected tickers
        
        mybar = self.ibkrbar_to_mybar(bar)
        curbar = self.id_current_bar[req_id]
        if mybar["t"] != curbar["t"]:
            #* Previous current bar finished --> update rolling features
            self.process_bar(req_id, curbar, True)
            if self.sort_bar(req_id, curbar):
                self.id_rth_processed_bars[req_id].loc[len(self.id_rth_processed_bars[req_id])] = curbar
                prev, res, std = self.app.query_model(
                    self.id_rth_processed_bars[req_id],
                    self.bar_width[req_id%2]
                )
                self.id_pred_bars[req_id] = {
                    "prev": prev,
                    "res": res,
                    "std": std
                }
        self.id_current_bar[req_id] = mybar

    def del_req_id_data(self, req_id):
        self.id_eth_bars.pop(req_id, None)
        self.id_rth_bars.pop(req_id, None)
        self.id_current_bar.pop(req_id, None)
        self.id_rolling_features.pop(req_id, None)
        self.id_rth_processed_bars.pop(req_id, None)
        self.id_pred_bars.pop(req_id, None)

    def remove_ticker(self, base_id):
        self.del_req_id_data(base_id)
        self.del_req_id_data(base_id+1)

class DataProcessor():
    def __init__(self):
        pass

    def fill_history(mybars, bar_width):
        df = pd.DataFrame(mybars).sort_values("t").reset_index(drop=True)
        now = datetime.now(ZoneInfo("America/New_York"))
        start_date = (pd.to_datetime(df["t"].iloc[0], unit="s", utc=True
                                     ).tz_convert("America/New_York").date())
        end_date = (pd.to_datetime(df["t"].iloc[-1], unit="s", utc=True
                                   ).tz_convert("America/New_York").date())
        expected_t = get_unix_timestamps(pd.date_range(start_date, end_date), bar_width)

        expected_t = expected_t // 1000
        expected_t = expected_t - bar_width * 60
        now_s = int(now.timestamp())
        expected_t = expected_t[(expected_t >= df["t"].iloc[0]) & (expected_t <= now_s)]
        df = (df.set_index("t").reindex(expected_t).rename_axis("t").reset_index())

        df["date"] = (pd.to_datetime(df["t"], unit="s", utc=True
                                     ).dt.tz_convert("America/New_York").dt.date)
        f_mask = df["c"].isna()
        df["f"] = f_mask.astype(int)
        df.loc[f_mask, ['v', 'n']] = 0
        c_ff = df["c"].ffill()
        for col in ["o", "h", "l", "c", "vw"]:
            df[col] = df[col].fillna(c_ff)
        df = df[df["c"].notna()].reset_index(drop=True)
        return df

    def process_history(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("t").copy()
        df["av"] = df["v"].rolling(90).mean()
        df["rv"] = df["v"] / df["av"]
        df["ema9"] = df["c"].ewm(span=9, adjust=False).mean()
        df["ema20"] = df["c"].ewm(span=20, adjust=False).mean()
        ema12 = df["c"].ewm(span=12, adjust=False).mean()
        ema26 = df["c"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["bar"] = df.groupby("date").cumcount() + 1
        return df

    def process_data(data: list, bar_width):
        df = DataProcessor.fill_history(data, bar_width)
        return DataProcessor.process_history(df)
          
if __name__ == "__main__":
    app = App()
    app.execute()