from fastapi import FastAPI

from model_query import query_model
api = FastAPI()

@api.get('/update/alltickers')
def update_all():
    """
    Recurring function call to highlight potential profitable tickers across the whole market
    """
    pass


@api.get('/update/{ticker}')
def update_ticker():
    """
    Loads history of current active ticker
    """
    pass