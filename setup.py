import pandas_market_calendars as mcal

DEBUG = True
API_KEY = "sdpbiDy3nfhuvQX2SBBtL6Gt2dl88ZrU"
YEAR_START = "2021"
YEAR_END = "2026"
DATE_RANGE = mcal.get_calendar("NYSE").schedule(f"{YEAR_START}-08-01",f"{YEAR_END}-12-31").index

AVG_VOLUME_PERIOD = 90                                                  # 
RV_THRESH = 0.75
MN = 1                                                                  # Pre-culls tickers for price
MX = 20     

ID = 1
PRESETS = {
    1: {
        "file_name": f"data_1min_{YEAR_START}_{YEAR_END}",
        "timeframe": "1/minute",
        "bar_per_day": 390,
        "bar_width": 1,
        "n_transformers": 4,
        "file_limit": 50
    },
    5: {
        "file_name": f"data_5min_{YEAR_START}_{YEAR_END}",
        "timeframe": "5/minute",
        "bar_per_day": 78,
        "bar_width": 5,
        "n_transformers": 4,
        "file_limit": 183
    },
}

FILE_LIMIT = PRESETS[ID]["file_limit"]
VERSION = f"{FILE_LIMIT}"

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
                  "n", "rv", "f"]
TARGET_FEATURES = ["o", "h", "l", "c"]

# Stock GPT -----------------------------------------------------------------------------------------
STEP = 1                            #! FUTURE HORIZON
SEQ_LEN = BAR_PER_DAY - STEP
OUTPUT_DIM = 256
DGF = 2.25                          #* Degrees of Freedom for Student-t

StockBPT_cfg = {
    "name": f"StockGPT-v{ID}{STEP}-{DGF}-{RV_THRESH}-{VERSION}",
    "checkpoint_path": f"model_parameters/checkpoint_stock_bpt_v{ID}{STEP}-{DGF}-{RV_THRESH}-{VERSION}",
    "best_path": f"model_parameters/best_stock_bpt_v{ID}{STEP}-{DGF}-{RV_THRESH}-{VERSION}",
    "input_features": INPUT_FEATURES,
    "target_features": TARGET_FEATURES,
    "bar_per_day": BAR_PER_DAY,
    "seq_len": SEQ_LEN,
    "output_dim": OUTPUT_DIM,
    "n_heads": 4,
    "n_transformers": PRESETS[ID]["n_transformers"],
    "qkv_bias": False,
    "step": STEP,
    "file_limit": PRESETS[ID]["file_limit"]
}

LinearModel_cfg = {
    "name": f"LinearModel-v{ID}{STEP}-{DGF}-{RV_THRESH}-{VERSION}",
    "checkpoint_path": f"model_parameters/checkpoint_linear_model_v{ID}{STEP}-{DGF}-{RV_THRESH}-{VERSION}",
    "best_path": f"model_parameters/best_linear_model_v{ID}{STEP}-{DGF}-{RV_THRESH}-{VERSION}",
    "input_features": INPUT_FEATURES,
    "target_features": TARGET_FEATURES,
    "seq_len": SEQ_LEN,
    "output_dim": OUTPUT_DIM,
    "step": STEP,
    "file_limit": PRESETS[ID]["file_limit"]
}

NaiveModel_cfg = {
    "name": f"NaiveModel-B{PRESETS[ID]['bar_width']}_{VERSION}",
    "input_features": INPUT_FEATURES,
    "target_features": TARGET_FEATURES,
    "file_limit": PRESETS[ID]["file_limit"]
}

path_test_2026 = f"preprocessed_data/{PRESETS[ID]['file_name'][:9]}_2026_2026_{VERSION}"
# Model Query -----------------------------------------------------------------------------------------

#path_stockBPT_B1 = "model_parameters/best_stock_bpt_1min"
#path_stockBPT_B5 = "model_parameters/best_stock_bpt_5min"
path_stockBPT_B1r = f"model_parameters/best_stock_bpt_B1r_{VERSION}"
path_stockBPT_B5r = f"model_parameters/best_stock_bpt_B5r_{VERSION}"