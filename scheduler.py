from controllers.unixtime import get_time_NY, get_time_SG, fmt
from controllers.ticker import Ticker
from controllers.mongo import MONGORECS

from apscheduler.schedulers.background import BlockingScheduler

print(f"Current time SG: {get_time_SG()}")
print(f"Current time NY: {get_time_NY()}")

def main_job(tickers):
    now = get_time_NY().strftime(fmt)
    now_sg = get_time_SG().strftime(fmt)
    recs = []
    for t in tickers:
        c, p, tn = t.get_pred()
        if p > c:
            d = {'symbol': tn, 'targetPrice': float(p), 'currentPrice': float(c)}
            recs.append(d)
    # upload recs to mongo
    to_up = {'us-date': now, 'sg-date': now_sg, 'recommendations': recs}
    MONGORECS.insert_one(to_up)
    print(f"Job done at SG time: {now_sg}. NY time: {now}")

TICKERS = ['AAPL', 'AMZN', 'BABA', 'BLK', 'BX', 'C', 'GOOG', 'NOK', 'NVDA', 'SHOP', 'TLT']
BEST = {'AAPL': 4, 'AMZN': 6, 'BABA': 6, 'BLK': 6, 'BX': 7, 'C': 3, 'GOOG': 6, 'NOK': 3, 'NVDA': 3, 'SHOP': 0, 'TLT': 5}

def test():
    tkrs = [Ticker(t, BEST[t]) for t in TICKERS]
    print('loaded models')
    main_job(tkrs)

def main():
    tkrs = [Ticker(t, BEST[t]) for t in TICKERS]
    print('loaded models')

    sched = BlockingScheduler(timezone="America/New_York")                      
    sched.add_job(main_job, trigger='cron', day_of_week='mon,tue,wed,thu,fri', hour='9-16', minute='59', args=[tkrs])
    # sched.add_job(main_job, trigger='cron',  hour='*', minute='59', args=[tkrs])
    sched.start()

"""
DST is active in US from:
Second Sunday of March and ends on the first Sunday of November.
"""

if __name__ == '__main__':
    test()
    # main()