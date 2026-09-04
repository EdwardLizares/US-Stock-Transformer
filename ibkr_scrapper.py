import threading
import tempfile
import pandas as pd
import pyarrow as pa

from pathlib import Path
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

from setup import BAR_WIDTH, INPUT_FEATURES
from data_filler import fill_file
from data_preprocessor import engineer_data

PORT = 7496
class IBKR(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.ready = threading.Event()
        self.events = {}
        self.bars = {}
        self.errors = {}

    def nextValidId(self, orderId):
        self.ready.set()

    def historicalData(self, reqId, bar):
        self.bars.setdefault(reqId, []).append(bar)

    def historicalDataEnd(self, reqId, start, end):
        if reqId in self.events:
            self.events[reqId].set()

    def error(self, reqId, errorTime: int, errorCode: int,
              errorString: str, advancedOrderRejectJson = ""):
        # Ignore informational connection messages.
        if errorCode not in (2104, 2106, 2158):
            self.errors[reqId] = (errorCode, errorString)
            if reqId in self.events:
                self.events[reqId].set()


def _stock_contract(ticker):
    contract = Contract()
    contract.symbol = ticker.upper()
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract


def _connect():
    ib = IBKR()
    ib.connect("127.0.0.1", PORT, 0)

    thread = threading.Thread(target=ib.run, daemon=True)
    thread.start()

    if not ib.ready.wait(10):
        ib.disconnect()
        raise TimeoutError("IBKR connection timed out.")

    return ib


def _historical_request(
    ib,
    req_id,
    ticker,
    end,
    duration,
    bar_size,
    use_rth=0,
    format_date=2,
):
    event = threading.Event()
    ib.events[req_id] = event
    ib.bars[req_id] = []

    ib.reqHistoricalData(
        req_id,
        _stock_contract(ticker),
        end,
        duration,
        bar_size,
        "TRADES",
        use_rth,
        format_date,
        False,
        [],
    )

    if not event.wait(30):
        ib.cancelHistoricalData(req_id)
        raise TimeoutError(f"Historical request {req_id} timed out.")

    if req_id in ib.errors:
        code, message = ib.errors.pop(req_id)
        raise RuntimeError(f"IBKR {code}: {message}")

    return ib.bars.pop(req_id)


# ----------------------------------------------------------------------
# 1. FETCH ONE DAY OF 1-MINUTE IBKR DATA
# ----------------------------------------------------------------------

def fetch_historical_data(ticker, date):
    date = pd.Timestamp(date)
    ib = _connect()

    try:
        end = f"{date.strftime('%Y%m%d')} 23:59:59 US/Eastern"

        bars = _historical_request(
            ib=ib,
            req_id=1,
            ticker=ticker,
            end=end,
            duration="7 D",
            bar_size=f"{BAR_WIDTH} min",
            use_rth=0,
            format_date=2,
        )
    finally:
        ib.disconnect()

    df = pd.DataFrame([
        {
            "v": float(bar.volume),
            "vw": float(bar.wap),
            "o": float(bar.open),
            "c": float(bar.close),
            "h": float(bar.high),
            "l": float(bar.low),
            "t": int(bar.date) * 1000,
            "n": int(bar.barCount),
            "T_1": ticker.upper(),
        }
        for bar in bars
    ])

    if df.empty:
        return df

    df = df.sort_values("t").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# 2. PREVIOUS 90 TRADING DAYS' AVERAGE DAILY VOLUME
# ----------------------------------------------------------------------

def fetch_avg_volume_90d(ticker, date):
    """
    Average full-day volume over the previous 90 trading days.

    The requested date itself is excluded.
    """

    date = pd.Timestamp(date)

    # Midnight at the START of the target date means the target session
    # cannot enter the historical window.
    end = f"{date.strftime('%Y%m%d')} 00:00:00 US/Eastern"

    ib = _connect()

    try:
        # 6 M gives plenty of room for >=90 actual trading sessions.
        bars = _historical_request(
            ib=ib,
            req_id=2,
            ticker=ticker,
            end=end,
            duration="5 M",
            bar_size="1 day",
            use_rth=0,
            format_date=1,
        )
    finally:
        ib.disconnect()

    daily = pd.DataFrame([
        {
            "date": pd.to_datetime(str(bar.date)),
            "volume": float(bar.volume),
        }
        for bar in bars
    ])

    daily = daily[daily["date"] < date].sort_values("date").tail(90)

    if len(daily) < 90:
        raise ValueError(
            f"Only {len(daily)} prior trading days returned for {ticker}; need 90."
        )

    return daily["volume"].mean()


# ----------------------------------------------------------------------
# 3. PRODUCE ONE 390-BAR MODEL-READY DATAFRAME
# ----------------------------------------------------------------------

def fetch_processed_data(ticker, date):
    date = pd.Timestamp(date)

    raw_df = fetch_historical_data(ticker, date)

    if raw_df.empty:
        raise ValueError(f"No IBKR data returned for {ticker} on {date.date()}.")

    avg_volume = fetch_avg_volume_90d(ticker, date)

    raw_dates = (
        pd.to_datetime(raw_df["t"], unit="ms", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.normalize()
        .dt.tz_localize(None)
        .drop_duplicates()
        .sort_values()
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        raw_folder = tmp / "raw"
        filled_folder = tmp / "filled"

        raw_folder.mkdir()
        filled_folder.mkdir()

        raw_path = raw_folder / f"{ticker.upper()}_{date:%Y-%m-%d}.parquet"
        raw_df.to_parquet(raw_path, index=False)

        fill_file(
            raw_path,
            filled_folder,
            pd.DatetimeIndex(raw_dates),
            store=True,
        )

        filled_path = filled_folder / raw_path.name
        df = pd.read_parquet(filled_path)

    # Engineer features using the warmup days too.
    df = engineer_data(df, None)

    # Keep only the requested day after rolling features/EMAs are calculated.
    df = df[df["date"] == date.date()].copy()
    df = df.sort_values("t").reset_index(drop=True)

    # Keep your original IBKR RV calculation.
    df["ibkr_rv"] = df["v"].cumsum() / avg_volume

    if len(df) != 390:
        raise ValueError(
            f"Expected 390 processed bars for {ticker} on {date.date()}, "
            f"got {len(df)}."
        )

    cols = [
        col
        for col in INPUT_FEATURES + ["Tk", "date", "ibkr_rv"]
        if col in df.columns
    ]

    df = df[cols]

    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")

    return df

def save_arrow(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, preserve_index=False)

    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

def handpick_data():
    ticker = input("Enter ticker: ").upper()
    date = input("Enter date (YYYY-MM-DD): ")

    df = fetch_processed_data(ticker, date)

    print(df)
    print("Shape:", df.shape)

    save_arrow(df, f"handpicked_data/preprocessed_data/{ticker}_{date}.arrow")

def handpick_data_by_date():
    tickers = []
    date = input("Enter date (YYYY-MM-DD): ")
    while True:
        ticker = input("Ticker: ").upper()
        if not ticker:
            break
        tickers.append(ticker)

    for ticker in tickers:
        df = fetch_processed_data(ticker, date)
        print(df)
        print("Shape:", df.shape)
        save_arrow(df, f"handpicked_data/test/{ticker}_{date}.arrow")

if __name__ == "__main__":
    handpick_data_by_date()