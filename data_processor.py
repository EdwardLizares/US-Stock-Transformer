from data_scrapper_per_ticker import data_scrapper
import pandas as pd 

# Cameron-Lizares Screener -> Volume > 5xAvg Volume, Last$ = 1-20, Gap% > 0
FILE_NAME = "data_day_2024"
AVG_VOLUME_DAYS = 90 #! testing, set to 90
RATIO_THRESHOLD = 5 #! testing, set to 5
HYPERPARAMETER_COMBOS = [["Ticker", "Date", "VWAP", "EMA9", "EMA20", "Open", "Close", "High", "Low", "NumTrades", "RelVolume", "Gap%", "Outcome"],
                         ["Ticker", "Date", "VWAP" "Open", "Low", "RelVolume", "Gap%", "Outcome"],
                         ["Ticker", "Date", "VWAP", "Open", "RelVolume", "Gap%", "Outcome"]]

def process_data(path_name: str) -> pd.DataFrame:
    df = pd.read_parquet(path_name)
    df = calculate_additional_hyperparameters(df)
    df = cameron_lizares_screener(df)
    df = select_hyperparameters(df, id=0) #! Incrase ID to lessen hyperparameters
    return df

def calculate_additional_hyperparameters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Ticker", "Date"])
    #* Add avg volume hyperparamter column
    df["AvgVolume"] = (
        df.groupby("Ticker")["Volume"].transform(
            lambda x: x.rolling(AVG_VOLUME_DAYS).mean()
            )
        )
    df["EMA9"] = (
        df.groupby("Ticker")["Close"].transform(
            lambda x: x.ewm(span=9, adjust=False).mean()
        )
    )
    df["EMA20"] = (
        df.groupby("Ticker")["Close"].transform(
            lambda x: x.ewm(span=20, adjust=False).mean()
        )
    )
    #* Add gap% hyperparameter
    df["Gap%"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1) * 100
    #* Add relative volume hyperparameter
    df["RelVolume"] = df["Volume"] / df["AvgVolume"]
    #* Add Outcome column (Clsoe / Open) as measure of success
    df["Outcome"] = (df["Open"].shift(-1) > df["Close"]).astype(int)

    #* Remove Nan 
    return df[(df["AvgVolume"].notna()) & 
              (df["Gap%"].notna()) & 
              (df["RelVolume"].notna()) & 
              (df["Outcome"].notna())]

def cameron_lizares_screener(df: pd.DataFrame) -> pd.DataFrame:
    #* Filter by ratio, close, gap% ...
    return df[(df["Volume"]>df["AvgVolume"]*RATIO_THRESHOLD) &
            (df["Close"].between(1, 20)) &
            (df["Gap%"]>0)]

def select_hyperparameters(df: pd.DataFrame, id: int) -> pd.DataFrame:
    return df[HYPERPARAMETER_COMBOS[id]]

if __name__ == "__main__":
    processed_data = process_data(f"raw_data/{FILE_NAME}.parquet")
    print(processed_data)
    processed_data.to_parquet(f"processed_data/{FILE_NAME}.parquet", index=False)