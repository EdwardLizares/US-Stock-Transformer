from datetime import date, datetime
from zoneinfo import ZoneInfo

from setup import path_app_state
from live_data_scrapper import add_ticker_predata

from setup import API_KEY, BAR_WIDTH

class AppState():
    def __init__(self, api_key: str, bar_width: int):
        self.api_key = api_key
        self.date = datetime.now(ZoneInfo("America/New_York")).date()
        self.bar_width = bar_width
        self.selected_tickers = set()

        self.data = DataState(self, "app_data")

    def get_apikey(self):
        return self.api_key
    
    def get_timeframe(self):
        return f"{self.bar_width}/minute"

    def get_date(self):
        return self.date
    
    def change_barwidth(self, bar_width):
        self.bar_width = bar_width

    def select_ticker(self, ticker: str):
        self.data.add_ticker(ticker)
        self.selected_tickers.add(ticker)

    def unselect_ticker(self, ticker: str):
        self.selected_tickers.remove(ticker)

class DataState():
    def __init__(self, app_state, output_path):
        self.app_state = app_state
        self.tickers = set()
        self.output_path = output_path #* Where data is stored for the app

    def add_ticker(self, ticker):
        if ticker not in self.tickers:
            self.tickers.add(ticker)
            add_ticker_predata(self.app_state, self.output_path, ticker)

if __name__ == "__main__":
    app = AppState(API_KEY, 15)
    app.data.add_ticker("TENX")