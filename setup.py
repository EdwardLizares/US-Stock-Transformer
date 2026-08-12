import pandas_market_calendars as mcal

API_KEY = "sdpbiDy3nfhuvQX2SBBtL6Gt2dl88ZrU"
DATE_RANGE = mcal.get_calendar("NYSE").schedule("2025-01-01","2025-12-31").index
AVG_VOLUME_PERIOD = 90                                              # 
RATIO_THRESHOLD = 5 #! testing, set to 5

# Data Scrapper ---------------------------------------------------------------------------------------
TIMEFRAME = "15/minute"                                                 # OHLC Bar Widths in url
MN = 1                                                                  # Pre-culls tickers for price
MX = 30                                                                 

path_raw_all_tickers = "raw_data/all_tickers"
path_raw_tickers_trimmed = "raw_data/all_tickers_trimmed_1_30"
path_data_scrapper = "raw_data/data_1hour_2025.parquet"                 #* Output path of data_scrapper.py

# Data Processor --------------------------------------------------------------------------------------   
BAR_WIDTH = 15                                                          # OHLC Bar Widths
HYPERPARAMETER_COMBOS = ["T", "t", "date", "vw", "ema9",                # Columns for processed data
                          "ema20", "o", "c", "h", "l", "n", 
                          "rv", "gp", "otc", "fb", "y"]

path_data_processor = "processed_data/data_15min_2025_filled.parquet"   #*Output path of data_procesor.py

# Dataset Builder --------------------------------------------------------------------------------------   
SEQ_LEN = 13

path_dataset_builder = "training_data/data_15min_2025_filled.parquet"
