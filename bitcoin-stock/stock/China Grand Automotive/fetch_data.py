import os
import urllib
import pandas as pd

# helpers
def get_date_col(df):
    '''
    Get date column
    '''
    for col in df.columns:
        try:
            if df[col].str.contains('-').all():
                return col
        except:
            continue

def sort_by_date(df):
    '''
    Sort datatime column
    '''
    date = get_date_col(df)
    df[date] = pd.to_datetime(df[date])
    return df.sort_values(by=date)


def fetch_stock(symbol):
    '''
    Symbol -- stock symbol
    Return -- stock data in the format of pandas dataframe in time sorted order
    '''
    api_key = os.environ['ALPHA_KEY']
    symbol = symbol

    url =  f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol={symbol}&apikey={api_key}&datatype=csv'

    response = urllib.request.urlopen(url)
    html = response.read()

    with open(f'data/{symbol}.csv', 'wb') as f:
            f.write(html)

    print('stock data fetching completed')
    print('reading data into dataframe')

    df = pd.read_csv(f'data/{symbol}.csv')

    return sort_by_date(df)
