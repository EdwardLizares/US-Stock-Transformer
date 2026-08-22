import torch

from datetime import date, datetime
from zoneinfo import ZoneInfo

from live_data_scrapper import add_ticker_predata
from model_query import setup_model, query_model, print_prediction

from setup import API_KEY, BAR_WIDTH, INPUT_FEATURES, TARGET_FEATURES
from setup import path_stockGPT_B1, path_stockGPT_B5
from setup import StockGPT_cfg

class AppState():
    def __init__(self):
        self.api_key = API_KEY
        self.date = datetime.now(ZoneInfo("America/New_York")).date()
        self.bar_width = BAR_WIDTH
        self.selected_tickers = set()
        self.input_features = INPUT_FEATURES
        self.target_features = TARGET_FEATURES

        #* MODELS -----------------------------
        self.stockGPT_B1, self.B1_device = setup_model(path_stockGPT_B1)
        self.stockGPT_B5, self.B5_device = setup_model(path_stockGPT_B5)

        #* DATA ------------------------------
        self.data = DataState(self, "app_data")

    def get_apikey(self):
        return self.api_key
    
    def get_timeframe(self):
        return f"{self.bar_width}/minute"

    def get_date(self):
        return self.date

    def get_inputfeatures(self):
        return self.input_features

    def get_targetfeatures(self):
        return self.target_features
    
    def change_barwidth(self, bar_width):
        self.bar_width = bar_width

    def select_ticker(self, ticker: str):
        self.data.add_ticker(ticker)

    def unselect_ticker(self, ticker: str):
        self.data.remove_ticker(ticker)

    #* Model Interaction Methods
    def query(self, model, ticker):
        if model == "B1":
            return query_model(self.stockGPT_B1, self.data.get_ticker_data(ticker))
        elif model == "B5":
            return query_model(self.stockGPT_B5, self.data.get_ticker_data(ticker))
        else:
            pass

class DataState():
    def __init__(self, app_state, output_path):
        self.app_state = app_state
        self.tickers = set()
        self.ticker_data = {}
        self.output_path = output_path #* Where data is stored for the app

    def add_ticker(self, ticker):
        if ticker not in self.tickers:
            self.tickers.add(ticker)
            self.ticker_data[ticker] = add_ticker_predata(self.app_state, self.output_path, ticker)

    def remove_ticker(self, ticker):
        if ticker in self.tickers:
            self.tickers.remove(ticker)
            del self.ticker_data[ticker]

    def get_ticker_data(self, ticker):
        df = self.ticker_data[ticker]
        latest_date = df["date"].max()
        df = df[df["date"] == latest_date]
        df = df[self.app_state.get_inputfeatures()]

        res = torch.tensor(df.to_numpy(), dtype=torch.float32).unsqueeze(0)
        #print(res.size())
        return res

import torch
if __name__ == "__main__":
    app = AppState()
    print("Welcome to the app ---------------------------\n")
    while (True):
        task = input((f"What would you like to do? (Enter a number)\n"
                      f"   1: Enter a new ticker to track\n"
                      f"   2: Remove a ticker from the tracking list\n"
                      f"   3: Ask the model to predict a ticker\n"
                      f"   q: Quit\n"
                      f"\n"
                      f"   "))
        if task == '3':
            ticker = input("Select a ticker: ")
            app.select_ticker(ticker)
            model = input("Select a model: ")
            prev, mean, std = app.query(model, ticker)
            print_prediction(prev, mean, std, app.input_features, app.target_features)
        if task == 'q':
            break

#* NOTES -- add functionality to store both 1m and 5m bars right now cannot change from 1m to 5m per ticker