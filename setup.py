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
COLUMNS = ["T", "date", "bar", "vw", "ema9",                    # Columns for processed data
           "ema20", "o", "c", "h", "l", "n", 
           "rv", "gp", "fb"]                                    #! Recent Change: Removed y

path_data_filler = f"filled_raw_data/{FILE_NAME}.parquet"               #*Output path of data_filler.py

# Data Preprocessor ------------------------------------------------------------------------------------
BAR_PER_DAY = 26

path_data_preprocessor = f"preprocessed_data/{FILE_NAME}.parquet"       #*Output path of data_preprocessor

# Dataloader Builder -------------------------------------------------------------------------------------- 

SPLIT = [0.75, 0.9, 1]
BATCH_SIZE = 256
INPUT_FEATURES = ["bar", "vw", "ema9", "ema20", "o", "c", "h", "l", 
                  "n", "rv", "gp", "f", "bf"]
TARGET_FEATURES = ["vw", "ema9", "ema20", "o", "c", "h", "l", 
                   "n", "rv", "gp"]

# Stock GPT -----------------------------------------------------------------------------------------
SEQ_LEN = 12
OUTPUT_DIM = 256

StockGPT_cfg = {
    "name": "stock_gpt_v1",
    "checkpoint_path": "model_parameters/checkpoint_stock_gpt_6M",
    "best_path": "model_parameters/best_stock_gpt_6M",
    "input_features": INPUT_FEATURES,
    "target_features": TARGET_FEATURES,
    "bar_per_day": BAR_PER_DAY,
    "seq_len": SEQ_LEN,
    "output_dim": OUTPUT_DIM,
    "n_heads": 4,
    "n_transformers": 4,
    "qkv_bias": False,
    "step": 4
}

LinearModel_cfg = {
    "name": "linear_model_cfg",
    "checkpoint_path": "model_parameters/checkpoint_linear_model_6K",
    "best_path": "model_parameters/best_linear_model_6K",
    "input_features": INPUT_FEATURES,
    "target_features": TARGET_FEATURES,
    "seq_len": SEQ_LEN,
    "output_dim": OUTPUT_DIM,
    "step": 2
}

NaiveModel_cfg = {
    "input_features": INPUT_FEATURES,
    "target_features": TARGET_FEATURES,
}