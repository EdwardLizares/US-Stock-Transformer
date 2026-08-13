import pandas_market_calendars as mcal

API_KEY = "sdpbiDy3nfhuvQX2SBBtL6Gt2dl88ZrU"
DATE_RANGE = mcal.get_calendar("NYSE").schedule("2025-01-01","2025-12-31").index
FILE_NAME = "data_15min_2025"

AVG_VOLUME_PERIOD = 90                                                  # 
RV_THRESH = 5
MN = 1                                                                  # Pre-culls tickers for price
MX = 30     

# Data Scrapper ----------------------------------------------------------------------------------------
TIMEFRAME = "15/minute"                                                 # OHLC Bar Widths in url                                                            

path_raw_all_tickers = "raw_data/all_tickers"
path_raw_tickers_trimmed = "raw_data/all_tickers_trimmed_1_30"
path_data_scrapper = f"raw_data/{FILE_NAME}.parquet"                    #* Output path of data_scrapper.py

# Data Filler ------------------------------------------------------------------------------------------
BAR_WIDTH = 15                                                          # OHLC Bar Widths
HYPERPARAMETERS = ["T", "date", "bar", "vw", "ema9",                    # Columns for processed data
                   "ema20", "o", "c", "h", "l", "n", 
                   "rv", "gp", "fb"]                                    #! Recent Change: Removed y

path_data_filler = f"filled_raw_data/{FILE_NAME}.parquet"               #*Output path of data_filler.py

# Data Preprocessor ------------------------------------------------------------------------------------
BAR_PER_DAY = 26

path_data_preprocessor = f"preprocessed_data/{FILE_NAME}.parquet"       #*Output path of data_preprocessor

# Dataloader Builder -------------------------------------------------------------------------------------- 

SPLIT = [75, 15, 10]
HYPERPARAMETERS = ["bar", "vw", "ema9", "ema20", "o", "c",                     #*Subset of total column
                   "h", "l", "n", "rv", "gp", "fb"],
# Transformer CFG

Stock_GPT_cfg = {
    "HYPERPARAMETERS": 12,
    "BAR_PER_DAY": BAR_PER_DAY,
    "BATCH_SIZE": 4,
    "SEQ_LEN": 25,
    "OUTPUT_DIM": 256,
    "N_HEADS": 8,
    "N_TRANSFORMERS": 12,
    "QKV_BIAS": False
}
