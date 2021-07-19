import torch
from torch import nn
import torch.nn.functional as F

import datetime as dt
import numpy as np
from joblib import load

import requests
import os

# model class
class LSTM_Model(nn.Module):
    def __init__(self, timesteps, n_features, lstm_size=64, fc_size=64):
        super().__init__()

        self.lstm = nn.LSTM(input_size=n_features, hidden_size=lstm_size, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features=timesteps*lstm_size, out_features=fc_size)
        self.fc2 = nn.Linear(in_features=fc_size, out_features=1)

    def forward(self, x):
        batch = x.size(0)
        x, _ = self.lstm(x)
        x = x.reshape(batch, -1)
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# get data from api
def get_api(ticker, api_key, num_pts):
    day_to = dt.datetime.now().date()
    day_from = day_to - dt.timedelta(days=7)

    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/{day_from}/{day_to}?unadjusted=false&sort=desc&limit=1000&apiKey={api_key}'

    res = requests.get(url)
    content = res.json()

    # get the most recent 16 data points
    results = content['results'][:num_pts][::-1]

    # save the most recent timestamp
    ts = results[-1]['t'] / 1000
    now = dt.datetime.fromtimestamp(ts)

    # the input format is OHCL
    actual_input = [[l['o'], l['h'], l['c'], l['l']] for l in results]

    raw_str = ' '.join([','.join(map(str, l)) for l in actual_input])
    return raw_str, now


# load the model
def load_model(ticker, mdl_num, timesteps=10, features=4):
    model = LSTM_Model(timesteps, features)
    model.load_state_dict(torch.load(f'models/{ticker}_{mdl_num}.pt', map_location='cpu'))
    model.eval()
    return model

# load the scaler
def load_scaler(ticker):
    scaler = load(f'scalers/{ticker}_scaler.joblib')
    return scaler

# convert from raw strings to floats
def get_input(raw_str):
    csv = raw_str.split(' ')
    inp = [list(map(float, line.split(','))) for line in csv]
    inp = np.array(inp).astype(np.float32)
    return inp.reshape(1, *inp.shape)

# scale inputs with a scaler
def scale_input(scaler, input):
    return scaler.transform(input.reshape(input.shape[0], -1)).reshape(input.shape)

# pass inputs into model to get N outputs (cause we need to compound)
def get_outputs(model, scaler, input, window=7, timesteps=10):
    outputs = []
    for i in range(window):
        new_inp = input[0][i:i+timesteps]
        new_inp = new_inp.reshape(1, *new_inp.shape)
        new_inp = scale_input(scaler, new_inp)
        new_inp = torch.from_numpy(new_inp)
        with torch.no_grad():
            out = model(new_inp)
            outputs.append(out.cpu().detach().flatten().item())
    return outputs

# get the last price (aka current price)
def get_ith_price(input, type, i=-1):
    t = {'open': 0, 'high': 1, 'close': 2, 'low': 3}[type]
    return input[0][i][t]

# get the moving average starting at index i
def get_moving_avg(input, window, type, i=0):
    prices = [get_ith_price(input, type, k) for k in range(i, i + window)]
    return sum(prices) / window

# get the prediction
def get_predicted_price(outputs, moving_avg):
    return sum(outputs) + moving_avg

# get the action
def get_predictions_preload(model, scaler, ticker, api_key, window=7, timesteps=10):
    raw_str, _ = get_api(ticker, api_key, window + timesteps - 1)
    actual_input = get_input(raw_str)
    # scaled_input = scale_input(scaler, actual_input)
    outputs = get_outputs(model, scaler, actual_input, window, timesteps)
    moving_avg = get_moving_avg(actual_input, window, 'close')
    pred = get_predicted_price(outputs, moving_avg)
    cur_price = get_ith_price(actual_input, 'close')
    return cur_price, pred

# FOLDER_NAME = 'future_moving_avg'
# TICKER = 'TLT'
# MODEL_NUM = 4
# WINDOW = 7
# TIMESTEPS = 10

