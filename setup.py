import pandas_market_calendars as mcal

API_KEY = "sdpbiDy3nfhuvQX2SBBtL6Gt2dl88ZrU"
DATE_RANGE = mcal.get_calendar("NYSE").schedule("2025-01-01","2025-12-31").index

AVG_VOLUME_PERIOD = 90                                                  # 
RV_THRESH = 5
MN = 1                                                                  # Pre-culls tickers for price
MX = 30     

ID = 5
PRESETS = {
    1: {
        "file_name": "data_1min_2025",
        "timeframe": "1/minute",
        "bar_per_day": 390,
        "bar_width": 1
    },
    2: {
        "file_name": "data_2min_2025",
        "timeframe": "2/minute",
        "bar_per_day": 195,
        "bar_width": 2
    },
    5: {
        "file_name": "data_5min_2025",
        "timeframe": "5/minute",
        "bar_per_day": 78,
        "bar_width": 5,
    },
    15: {
        "file_name": "data_15min_2025",
        "timeframe": "15/minute",
        "bar_per_day": 26,
        "bar_width": 15
    }
}

# Data Scrapper ----------------------------------------------------------------------------------------
TIMEFRAME = PRESETS[ID]["timeframe"]                                    # OHLC Bar Widths in url                                                            

path_raw_all_tickers = "raw_data/all_tickers"
path_raw_tickers_trimmed = "raw_data/all_tickers_trimmed_1_30"
path_data_scrapper = f"raw_data/{PRESETS[ID]['file_name']}"             #* Output path of data_scrapper.py

# Data Filler ------------------------------------------------------------------------------------------
BAR_WIDTH = PRESETS[ID]["bar_width"]                                    # OHLC Bar Widths in minutes
#COLUMNS = ["T_1", "date", "bar", "vw", "ema9",                         # Columns for processed data
#           "ema20", "o", "c", "h", "l", "n", 
#           "rv", "gp", "fb"]                                           #! Recent Change: Removed y

path_data_filler = f"filled_raw_data/{PRESETS[ID]['file_name']}"        #*Output path of data_filler_db.py

# Data Preprocessor ------------------------------------------------------------------------------------
BAR_PER_DAY = PRESETS[ID]["bar_per_day"]

path_data_preprocessor = f"preprocessed_data/{PRESETS[ID]['file_name']}"       #*Output path of data_preprocessor

# Dataloader Builder -------------------------------------------------------------------------------------- 

SPLIT = [0.75, 0.9, 1]
BATCH_SIZE = 256
NUM_WORKERS = 2
PERSISTENT_WORKERS = True
INPUT_FEATURES = ["bar", "vw", "ema9", "ema20", "macd", "o", "h", "l", "c",
                  "n", "rv", "f", "fb"]
TARGET_FEATURES = ["o", "h", "l", "c"]
FILE_LIMIT = 10

# Stock GPT -----------------------------------------------------------------------------------------
SEQ_LEN = BAR_PER_DAY - 1
STEP = 1
OUTPUT_DIM = 256

StockGPT_cfg = {
    "name": f"StockGPT-B{PRESETS[ID]['bar_width']}",
    "checkpoint_path": f"model_parameters/checkpoint_stock_gpt_{ID}min",
    "best_path": f"model_parameters/best_stock_gpt_{ID}min",
    "input_features": INPUT_FEATURES,
    "target_features": TARGET_FEATURES,
    "bar_per_day": BAR_PER_DAY,
    "seq_len": SEQ_LEN,
    "output_dim": OUTPUT_DIM,
    "n_heads": 4,
    "n_transformers": 4,
    "qkv_bias": False,
    "step": STEP,
    "file_limit": FILE_LIMIT
}

LinearModel_cfg = {
    "name": f"LinearModel-B{PRESETS[ID]['bar_width']}",
    "checkpoint_path": f"model_parameters/checkpoint_linear_model_{ID}min",
    "best_path": f"model_parameters/best_linear_model_{ID}min",
    "input_features": INPUT_FEATURES,
    "target_features": TARGET_FEATURES,
    "seq_len": SEQ_LEN,
    "output_dim": OUTPUT_DIM,
    "step": STEP,
    "file_limit": FILE_LIMIT
}

NaiveModel_cfg = {
    "name": f"NaiveModel-B{PRESETS[ID]['bar_width']}",
    "input_features": INPUT_FEATURES,
    "target_features": TARGET_FEATURES,
}