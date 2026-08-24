from ibapi.client import *
from ibapi.wrapper import *
import time
import threading
from ibapi.ticktype import TickTypeEnum
from setup import DEBUG

PORT = 7496
class IBKRClient(EClient, EWrapper):
    def __init__(self, app):
        EClient.__init__(self, self)
        self.orderId = -2   #! HARDCODED
        self.tickers = set()
        self.ticker_ids = {} #* ticker -> req_id
        self._id_tickers = {} #* req_id -> ticker
        self.ticker_contracts = {} #* ticker -> contract

        self.app = app

        self.connect("127.0.0.1", PORT, 0)
        self.thread = threading.Thread(target=self.run,)
        self.thread.start()
        time.sleep(1)
        self.reqMarketDataType(3) #* 3: Delayed, 1: Live
        if DEBUG:
            print("Client initialized")

    def error(self, reqId: TickerId, errorTime: int, errorCode: int,
              errorString: str, advancedOrderRejectJson = ""):
        if DEBUG:
            print("Error. Id:", reqId, errorTime, "Code:", errorCode, "Msg:", errorString,
                  "AdvancedOrderRejectJson:", advancedOrderRejectJson)
 
    def close(self):
        if self.isConnected():
            self.disconnect()
        if self.thread.is_alive():
            self.thread.join(timeout=2)

    def nextId(self):
        self.orderId += 2 #! HARDCODED: 1min and 5min
        return self.orderId

    def historicalData(self, req_id, bar):
        """
        Initial callback from ibkr after subscribing to a ticker
        """
        if DEBUG:
            print("historicalData callback received")
        self.app.data.initialize_id_history(req_id, bar)

    def historicalDataEnd(self, req_id, start, end):
        # initial history is now fully loaded
        if DEBUG:
            print("historicalDataEnd callback received")
        self.app.data.finish_id_initialization(req_id)

    def historicalDataUpdate(self, req_id, bar):
        # latest bar being updated live
        self.app.data.update_id_data(req_id, bar)

    def request_historical_data(self, req_id, contract):
        """
        Requests historical data for all time frames
        """
        self.reqHistoricalData(
            req_id, contract, "", "27000 S", "1 min", "TRADES", 1, 2, True, []
        )
        self.reqHistoricalData(
            req_id+1, contract, "", "48600 S", "5 mins", "TRADES", 1, 2, True, []
        )
        if DEBUG:
            print(f"Requested historical data for base_id:{req_id}")

    def unsubscribe(self, ticker):
        req_id = self.ticker_ids[ticker]
        self.cancelHistoricalData(req_id)
        self.cancelHistoricalData(req_id + 1)

    @staticmethod
    def contract(ticker):
        """
        Returns an ibapi.contract object for the given ticker
        Functions include 
        """
        contract = Contract()
        contract.symbol = ticker
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract

    def set_id_ticker(self, req_id, ticker):
        self._id_tickers[req_id//2] = ticker

    def get_id_ticker(self, req_id):
        return self._id_tickers[req_id//2]

    def del_id_tickers(self, req_id):
        del self._id_tickers[req_id//2]

    def add_ticker(self, ticker):
        if ticker not in self.tickers:
            contract = self.contract(ticker)
            req_id = self.nextId()

            self.tickers.add(ticker)
            self.ticker_ids[ticker] = req_id            #* Base id
            self.set_id_ticker(req_id, ticker)
            self.ticker_contracts[ticker] = contract
            if DEBUG:
                print(f"Requesting historical data for {ticker}")
            self.request_historical_data(req_id, contract)

    def remove_ticker(self, ticker):
        if ticker in self.tickers:
            self.unsubscribe(ticker)
            self.tickers.remove(ticker)
            base_id = self.ticker_ids[ticker]
            self.del_id_tickers(base_id)
            del self.ticker_ids[ticker]
            del self.ticker_contracts[ticker]

            return base_id
        return -1