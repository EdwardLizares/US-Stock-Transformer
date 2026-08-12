import pandas_market_calendars as mcal

API_KEY = "sdpbiDy3nfhuvQX2SBBtL6Gt2dl88ZrU"
DATE_RANGE = mcal.get_calendar("NYSE").schedule("2025-01-01","2025-12-31").index
FILE_NAME = "data_15min_2025"

AVG_VOLUME_PERIOD = 90                                                  # 
RV_THRESH = 5
MN = 1                                                                  # Pre-culls tickers for price
MX = 30     

# Data Scrapper ---------------------------------------------------------------------------------------
TIMEFRAME = "15/minute"                                                 # OHLC Bar Widths in url                                                            

path_raw_all_tickers = "raw_data/all_tickers"
path_raw_tickers_trimmed = "raw_data/all_tickers_trimmed_1_30"
path_data_scrapper = f"raw_data/{FILE_NAME}.parquet"                    #* Output path of data_scrapper.py

# Data Filler --------------------------------------------------------------------------------------   
BAR_WIDTH = 15                                                          # OHLC Bar Widths
HYPERPARAMETERS = ["T", "date", "bar", "vw", "ema9",                    # Columns for processed data
                   "ema20", "o", "c", "h", "l", "n", 
                   "rv", "gp", "fb", "y"]

path_data_filler = f"filled_raw_data/{FILE_NAME}.parquet"               #*Output path of data_filler.py

# Data Preprocessor ------------------------------------------------------------------------------------
BAR_PER_DAY = 27

path_data_preprocessor = f"preprocessed_data/{FILE_NAME}.parquet"       #*Output path of data_preprocessor

# Dataset Builder --------------------------------------------------------------------------------------   
SEQ_LEN = 13