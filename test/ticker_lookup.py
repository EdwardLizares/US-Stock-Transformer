import pandas as pd
import requests
import time

API_KEY = "sdpbiDy3nfhuvQX2SBBtL6Gt2dl88ZrU"

def ticker_lookup(test_cases: list) -> list:
    results = []
    for [ticker, date] in test_cases:
        url = (
            f"https://api.massive.com/v2/aggs/ticker/"
            f"{ticker}/range/1/day/{date}/{date}?apiKey={API_KEY}"
        )
        try:
            response = requests.get(url)
            response.raise_for_status()
            resp = response.json()["results"]
        except requests.exceptions.RequestException as e:
            results.append(e)

        results.append(resp)
        time.sleep(10)

    return results

if __name__ == "__main__":
    test_cases = [("ZVSA", "2025-01-10"), ("ZSB", "2025-01-21")]
    print(ticker_lookup(test_cases))