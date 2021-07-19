from controllers.polygon import PolygonParams, POLYGON_KEY, polygon_agg_call, polygon_clean_json
from controllers.unixtime import get_time_NY, get_time_SG, fmt
from datetime import timedelta

from controllers.mongo import MONGOPOLY

import deps as dp

class Ticker:

    def __init__(self, ticker_name, mdl_num):
        # Static Attributes
        self.ticker = ticker_name
        self.model = dp.load_model(ticker_name, mdl_num)
        self.scaler = dp.load_scaler(ticker_name)

    def get_pred(self):
        cur, pred = dp.get_predictions_preload(self.model, self.scaler, self.ticker, POLYGON_KEY)
        return cur, pred, self.ticker

    def send_polygon_to_mongo(self):
        # placeholder method - delete later on
        params = PolygonParams( self.ticker, 
                                "1",
                                "hour", 
                                (get_time_NY() - timedelta(days=3)).strftime("%Y-%m-%d"),
                                get_time_NY().strftime("%Y-%m-%d"),
                                "false",
                                "asc",
                                "50000",
                                POLYGON_KEY)

        json_obj = polygon_agg_call(params)
        json_obj['timestamp_sg'] = get_time_SG().strftime(fmt)
        json_obj['timestamp_ny'] = get_time_NY().strftime(fmt)
        MONGOPOLY.insert_one(json_obj)

