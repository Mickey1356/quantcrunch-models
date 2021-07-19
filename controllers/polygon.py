import requests
from collections import namedtuple
from controllers.unixtime import unix_to_NY, fmt
from decouple import config
import datetime

POLYGON_KEY = config('POLYGON_KEY')
parameters = "ticker multiplier timespan from_date to_date unadjusted sort limit KEY"
PolygonParams = namedtuple("PolygonParams", parameters)

def polygon_agg_call(polyparams):

    url = "https://api.polygon.io/v2/aggs/ticker/" \
        + polyparams.ticker \
        + "/range/" \
        + polyparams.multiplier \
        + "/" \
        + polyparams.timespan \
        + "/" \
        + polyparams.from_date \
        + "/" \
        + polyparams.to_date \
        + "?unadjusted=" \
        + polyparams.unadjusted \
        + "&sort=" \
        + polyparams.sort \
        + "&limit=" \
        + polyparams.limit \
        + "&apiKey=" \
        + polyparams.KEY

    response = requests.request("GET", url)
    assert response.status_code == 200
    res = response.json()
    return res

def polygon_agg_convert_ts(json):
    """
    Convert timestamps in json response from unix to human-readable strings
    Return the full json with only the timestamps changed.
    Note that the json object is mutable, thats why I can modify it like that.
    """
    for bar in json['results']:
        bar['t'] = unix_to_NY(bar['t']).strftime(fmt)
    return json

def polygon_clean_json(json):
    """
    Extract only the high prices
    """
    results = json['results']
    return [ (bar['t'], bar['h']) for bar in results ]

def retrieve_nlag_returns_and_latest_price(time_price_arr, n_lag):
    """
    Returns 2 objects: (list, float)
    list: return list of return values; no. of elements: n_lag.
    list is in ascending order of time (i.e. more recent timestamps to the right)
    this fits what the model requires
    float: return 'latest price; is not literally the latest price. 
    To be exact, it refers to price that is in the most recent hour-end.
    For example, if now is 11:31AM, then latest price is that on 11:00AM.
    price = high_price of candlestick. 
    """
    assert len(time_price_arr) > n_lag 

    price_ls = []
    for ts, price in time_price_arr:
        ts = datetime.datetime.strptime(ts, fmt).time() # convert string to dt.time()
        if datetime.time(9) <= ts < datetime.time(16): # filter prices whereby ts is between 9am to 4pm, NY time
            price_ls.append(price)
    
    ret_ls = []
    for i in range(len(price_ls)- 1):
        ret = float(price_ls[i+1]) / float(price_ls[i]) - 1
        ret *= 100
        ret_ls.append(ret)

    return ret_ls[-n_lag:], price_ls[-1]