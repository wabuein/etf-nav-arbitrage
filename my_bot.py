from quant_edge_lab import run, Market, TICKERS
import numpy as np

COST_BP = 32

class MyBot:

    def __init__(self):
        self.cooldown = 0

    def on_tick(self, v):
        # finding net value of assets
        nav = np.dot(v.stock_mid, v.shares_per_unit)

        # finding the gap between the etf mid and the net value of assets
        gap_bp = ((v.etf_mid / nav) - 1) * 10000

        # finding the commission/fee bp
        etf_half_bp = (v.etf_ask - v.etf_bid)/2
        etf_half_bp = etf_half_bp/v.etf_mid * 10000

        # setting direction
        direction = 0

        if (gap_bp > 0):
            direction = -1
        elif gap_bp < 0:
            direction = 1

        # calculating possible trade gaps 
        cost_bp = COST_BP
        edge_bp = max(0, (abs(gap_bp) - cost_bp))

        # calculating confidence
        full_edge_bp = edge_bp / 10
        confidence = min(1, full_edge_bp)   

        #setting portfolio size        
        max_dollars = 3_000_000

        #settings the amount of units we'd like to buy
        units = round(confidence * max_dollars/v.etf_mid)

        #setting the target
        target = units * direction

        # making sure that the edge is still valid and should not be closed
        if (v.etf_pos != 0 and abs(gap_bp) > 2):
            if (direction * v.etf_pos >= 0):
                target = v.etf_pos
            
        # making sure that we end the trade before the market closes
        if (v.ticks_left < 30):
            target = 0

        # calculating the units we'd like to trade
        etf_trade = target - v.etf_pos

        # calculating the equal amount of units we'd like to trade against the EFT
        stock_trade = -etf_trade * v.shares_per_unit

        
       ### if v.t % 500 == 0 or v.ticks_left == 20:
       ###     print(v.t, gap_bp, str(direction), edge_bp, confidence, target, etf_trade
       ###           , stock_trade

        orders = [Market("ETF", etf_trade)]
        for i in range(len(TICKERS)):
            orders.append(Market(TICKERS[i], stock_trade[i]))
        return orders

for cost in [16, 24, 32, 40, 48]:
    COST_BP = cost

    results =[]
    max_profit=0

    for seed in range(1,30):
        bot = MyBot()
        result = run(bot, seed=seed)
        max_profit+=result["pnl"]
        results.append(result)

    print(f"cost: {cost} max profit: {max_profit}")