from ibapi.client import *
from ibapi.wrapper import *
import time
import threading
from ibapi.ticktype import TickTypeEnum

# Default Ports:
# TWS Live Account: 7946
# TWS Paper Account: 7947
# IB Gateway Live Account: 4001
# IB Gateway Paper Account: 4002
PORT = 7496
class IBKRClient(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        self.connect("127.0.0.1", PORT, 0)

        self.thread = threading.Thread(
            target=self.run,
        )
        self.thread.start()
        time.sleep(1)
        self.reqMarketDataType(3)

        self.orderId = 0
        self.tickers = set()
        self.ticker_ids = {} #* ticker, req_id
        self.ticker_contracts = {} #* ticker, contract

    def close(self):
        if self.isConnected():
            self.disconnect()
        if self.thread.is_alive():
            self.thread.join(timeout=2)

    def nextId(self):
        self.orderId += 1
        return self.orderId
    
    def tickPrice(self, reqId, tickType, price, attrib):
        print(
            f"reqId: {reqId}, tickType: {TickTypeEnum.toStr(tickType)}, price: {price}, attrib: {attrib}"
        )

    def tickSize(self, reqId, tickType, size):
        print(f"reqId: {reqId}, tickType: {TickTypeEnum.toStr(tickType)}, size: {size}")

    def set_request_data_type(self, id: int):
        self.reqMarketDataType(id)

    def subscribe(self, ticker):
        """
        Returns the tickers associated id
        """
        contract = self.contract(ticker)
        req_id = self.nextId()
        self.reqMktData(req_id, contract, "", False, False, [])
        return contract, req_id

    def unsubscribe(self, ticker):
        self.cancelMktData(self.ticker_ids[ticker])

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

    def add_ticker(self, ticker):
        if ticker not in self.tickers:
            contract, req_id = self.subscribe(ticker)
            self.tickers.add(ticker)
            self.ticker_ids[ticker] = id
            self.ticker_contracts[ticker] = contract

    def remove_ticker(self, ticker):
        if ticker in self.tickers:
            self.unsubscribe(ticker)
            self.tickers.remove(ticker)
            del self.ticker_ids[ticker]
            del self.ticker_contracts[ticker]

if __name__ == "__main__":
    client = IBKRClient()
    try:
        client.add_ticker("TENX")
    except:
        client.close()