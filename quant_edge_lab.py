"""
quant_edge_lab.py  -  ETF arbitrage & market-making lab
=======================================================
Practice for the JPMorganChase x AmplifyME Quant Edge x Trading Challenge.

What this simulates (the things a "fill at mid" backtest hides):
  * bid/ask spreads on the ETF and each of its 7 stocks, wider in volatile spells
  * finite depth: big market orders walk the book and pay more than the touch
  * latency: orders sent on tick t fill at tick t+1 prices  (-> legging risk)
  * stale quotes: sometimes one stock's displayed price stops updating
  * news shocks that jump the ETF premium and raise volatility for a while
  * passive limit orders that fill only when other traders' market orders arrive,
    partially, and more often when those traders know where price is going
    (adverse selection)
  * fees on every fill

Usage
  python quant_edge_lab.py                 # compare the 4 reference strategies (5 seeds)
  python quant_edge_lab.py --mine          # also run YOUR strategy (edit MyStrategy below)
  python quant_edge_lab.py --seeds 20      # more seeds = more honest averages
  python quant_edge_lab.py --plot          # save pnl.png (needs matplotlib)
  python quant_edge_lab.py --drill         # print the 15-minute rewrite drill

Strategy API (deliberately small, so you can map it onto the real platform):
  class S:
      def on_tick(self, v):            # v = View (see class View)
          return [Market("ETF", +100), Market("AAPL", -30), ...]
                 # or [Quote("ETF", bid_px, bid_qty, ask_px, ask_qty)]  (replaces your old quote)
                 # or [CancelQuotes("ETF")]
Needs: numpy.  Python 3.8+.
"""

import argparse
import math
from dataclasses import dataclass, field

import numpy as np

# =============================================================== instruments
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
START_PX = np.array([230.0, 510.0, 180.0, 220.0, 190.0, 720.0, 330.0])
SHARES_PER_UNIT = np.array([0.30, 0.15, 0.40, 0.35, 0.40, 0.10, 0.25])   # basket per 1 ETF unit
BETAS = np.array([1.0, 0.9, 1.6, 1.2, 1.0, 1.3, 1.9])                  # to the common factor

FEE = 0.00025            # 2.5bp of notional per fill  (~0.10% round trip on the ETF incl. spread)
STOCK_HALF_SPREAD = 0.00010   # 1bp each side of mid in calm markets
ETF_HALF_SPREAD = 0.00020     # 2bp
STOCK_DEPTH = 400_000    # $ notional at the touch before you start walking the book
ETF_DEPTH = 1_500_000


# =============================================================== order types
@dataclass
class Market:
    sym: str
    qty: float           # +buy / -sell, in shares (ETF units for "ETF")


@dataclass
class Quote:
    sym: str             # only "ETF" supports passive quotes in this lab
    bid_px: float
    bid_qty: float
    ask_px: float
    ask_qty: float


@dataclass
class CancelQuotes:
    sym: str


# =============================================================== the market
class Market_Data:
    """Generates the 'true' path. Strategies only ever see the displayed (possibly stale) quotes."""

    def __init__(self, n_ticks=4000, seed=0):
        rng = np.random.default_rng(seed)
        self.n = n_ticks
        k = len(TICKERS)

        # volatility regime: news raises vol, which decays back
        news = rng.random(n_ticks) < 0.006
        volmult = np.ones(n_ticks)
        for t in range(1, n_ticks):
            volmult[t] = 1 + (volmult[t - 1] - 1) * 0.985 + (2.5 if news[t] else 0)
        self.news = news
        self.volmult = volmult

        f = rng.normal(0, 0.0007, n_ticks) * volmult
        f[news] += rng.choice([-1, 1], news.sum()) * rng.uniform(0.003, 0.008, news.sum())
        idio = rng.normal(0, 0.0008, (n_ticks, k)) * volmult[:, None]
        rets = f[:, None] * BETAS + idio
        self.stock_mid = START_PX * np.exp(np.cumsum(rets, axis=0))
        self.nav = self.stock_mid @ SHARES_PER_UNIT
        self.factor = f

        # ETF premium: mean-reverting with jumps on news and random flow shocks
        prem = np.zeros(n_ticks)
        for t in range(1, n_ticks):
            shock = rng.normal(0, 0.00045) * volmult[t]
            if news[t]:
                shock += rng.choice([-1, 1]) * rng.uniform(0.002, 0.006)
            elif rng.random() < 0.008:
                shock += rng.choice([-1, 1]) * rng.uniform(0.0015, 0.004)
            prem[t] = prem[t - 1] * 0.965 + shock
        self.prem = prem
        self.etf_mid = self.nav * (1 + prem)

        # displayed stock prices: occasionally one goes stale for a few ticks
        disp = self.stock_mid.copy()
        t = 1
        while t < n_ticks:
            if rng.random() < 0.004:
                i = rng.integers(k)
                L = rng.integers(2, 8)
                disp[t:t + L, i] = disp[t - 1, i]
                t += L
            t += 1
        self.stock_disp = disp

        # other traders' market orders that can hit passive quotes on the ETF
        self.arrivals = rng.poisson(1.2, n_ticks)
        self.rng = rng

    def half_spread(self, t, base):
        return base * (1 + 1.5 * (self.volmult[t] - 1) / 2.5)


@dataclass
class View:
    t: int
    ticks_left: int
    etf_bid: float
    etf_ask: float
    stock_bid: np.ndarray
    stock_ask: np.ndarray
    etf_pos: float
    stock_pos: np.ndarray
    cash: float
    news: bool
    shares_per_unit: np.ndarray = field(default_factory=lambda: SHARES_PER_UNIT)

    @property
    def etf_mid(self):
        return (self.etf_bid + self.etf_ask) / 2

    @property
    def stock_mid(self):
        return (self.stock_bid + self.stock_ask) / 2

    @property
    def nav(self):
        return float(self.stock_mid @ self.shares_per_unit)


# =============================================================== the exchange
class Exchange:
    def __init__(self, md: Market_Data):
        self.md = md
        self.etf_pos = 0.0
        self.stock_pos = np.zeros(len(TICKERS))
        self.cash = 0.0
        self.fees = 0.0
        self.spread_paid = 0.0       # cost vs mid of aggressive fills
        self.fills = 0
        self.quote = None            # resting ETF quote
        self.pending = []            # market orders waiting one tick (latency)
        self.passive_fills = 0

    # ---- prices -------------------------------------------------------------
    def etf_quotes(self, t):
        hs = self.md.half_spread(t, ETF_HALF_SPREAD)
        m = self.md.etf_mid[t]
        return m * (1 - hs), m * (1 + hs)

    def stock_quotes(self, t, displayed=True):
        hs = self.md.half_spread(t, STOCK_HALF_SPREAD)
        m = self.md.stock_disp[t] if displayed else self.md.stock_mid[t]
        return m * (1 - hs), m * (1 + hs)

    # ---- execution ----------------------------------------------------------
    def _aggress(self, mid, half_spread, depth, qty):
        """Market order: pay the half-spread, plus extra if you're bigger than the touch."""
        notional = abs(qty) * mid
        walk = max(0.0, notional / depth - 1) * 0.5           # each extra 'depth' costs another half-spread/2
        px = mid * (1 + np.sign(qty) * half_spread * (1 + walk))
        return px

    def _book(self, qty, px):
        self.cash -= qty * px
        fee = abs(qty) * px * FEE
        self.cash -= fee
        self.fees += fee
        self.fills += 1

    def execute_pending(self, t):
        md = self.md
        for o in self.pending:
            if abs(o.qty) < 1e-9:
                continue
            if o.sym == "ETF":
                mid = md.etf_mid[t]
                px = self._aggress(mid, md.half_spread(t, ETF_HALF_SPREAD), ETF_DEPTH, o.qty)
                self.etf_pos += o.qty
            else:
                i = TICKERS.index(o.sym)
                mid = md.stock_mid[t, i]                          # you fill at the TRUE price
                px = self._aggress(mid, md.half_spread(t, STOCK_HALF_SPREAD), STOCK_DEPTH, o.qty)
                self.stock_pos[i] += o.qty
            self.spread_paid += abs(px - mid) * abs(o.qty)
            self._book(o.qty, px)
        self.pending = []

    def match_passive(self, t):
        """Other traders' market orders arrive and may hit your resting ETF quote."""
        if not self.quote:
            return
        md, q = self.md, self.quote
        bid_touch, ask_touch = self.etf_quotes(t)
        nxt = md.etf_mid[min(t + 1, md.n - 1)] - md.etf_mid[t]   # informed traders know this
        for _ in range(md.arrivals[t]):
            informed = md.rng.random() < 0.35
            if informed:
                side = "sell" if nxt < 0 else "buy"
            else:
                side = "sell" if md.rng.random() < 0.5 else "buy"
            size = md.rng.exponential(1500)                       # ETF units
            if side == "sell" and q.bid_qty > 0 and q.bid_px >= bid_touch * (1 - 1e-12):
                prio = 1.0 if q.bid_px > bid_touch else 0.5       # inside the spread = you're first
                fill = min(q.bid_qty, size * prio)
                self.etf_pos += fill
                q.bid_qty -= fill
                self._book(fill, q.bid_px)
                self.passive_fills += 1
            elif side == "buy" and q.ask_qty > 0 and q.ask_px <= ask_touch * (1 + 1e-12):
                prio = 1.0 if q.ask_px < ask_touch else 0.5
                fill = min(q.ask_qty, size * prio)
                self.etf_pos -= fill
                q.ask_qty -= fill
                self._book(-fill, q.ask_px)
                self.passive_fills += 1

    def submit(self, orders):
        for o in orders or []:
            if isinstance(o, Market):
                self.pending.append(o)
            elif isinstance(o, Quote):
                self.quote = Quote(o.sym, o.bid_px, max(0, o.bid_qty), o.ask_px, max(0, o.ask_qty))
            elif isinstance(o, CancelQuotes):
                self.quote = None

    def mark(self, t):
        return self.cash + self.etf_pos * self.md.etf_mid[t] + self.stock_pos @ self.md.stock_mid[t]

    def net_exposure(self, t):
        return self.etf_pos * self.md.etf_mid[t] + self.stock_pos @ self.md.stock_mid[t]

    def beta_exposure(self, t):
        """$ exposure to the common factor (ETF beta = basket-weighted beta)."""
        w = SHARES_PER_UNIT * self.md.stock_mid[t]
        etf_beta = (w @ BETAS) / w.sum()
        return self.etf_pos * self.md.etf_mid[t] * etf_beta + (self.stock_pos * self.md.stock_mid[t]) @ BETAS


# =============================================================== run loop
def run(strategy, seed=0, n_ticks=4000, record=False):
    md = Market_Data(n_ticks, seed)
    ex = Exchange(md)
    pnl_path, beta_pnl, nets = [], 0.0, []
    for t in range(n_ticks):
        ex.execute_pending(t)                  # orders from t-1 fill now (latency)
        ex.match_passive(t)
        if t > 0:
            beta_pnl += ex.beta_exposure(t - 1) * md.factor[t]   # PnL explained by the market move
        eb, ea = ex.etf_quotes(t)
        sb, sa = ex.stock_quotes(t, displayed=True)
        v = View(t, n_ticks - 1 - t, eb, ea, sb, sa, ex.etf_pos, ex.stock_pos.copy(), ex.cash, bool(md.news[t]))
        if t < n_ticks - 2:
            ex.submit(strategy.on_tick(v))
        elif t == n_ticks - 2:                 # the bell: everything is flattened at market
            ex.quote = None
            ex.submit([Market("ETF", -ex.etf_pos)] +
                      [Market(s, -ex.stock_pos[i]) for i, s in enumerate(TICKERS)])
        pnl_path.append(ex.mark(t))
        nets.append(abs(ex.net_exposure(t)))
    pnl_path = np.array(pnl_path)
    d = np.diff(pnl_path)
    sharpe = d.mean() / d.std() * math.sqrt(len(d)) if d.std() > 0 else 0.0
    dd = float(np.max(np.maximum.accumulate(pnl_path) - pnl_path))
    out = dict(pnl=pnl_path[-1], fees=ex.fees, spread=ex.spread_paid, fills=ex.fills,
               passive=ex.passive_fills, sharpe=sharpe, maxdd=dd,
               avg_net=float(np.mean(nets)), beta_pnl=beta_pnl)
    if record:
        out["path"] = pnl_path
    return out


# =============================================================== reference strategies
def basket_orders(d_units, w=SHARES_PER_UNIT):
    """Hedge d_units of ETF with the opposite basket. Whole shares only."""
    return [Market(s, -round(d_units * w[i])) for i, s in enumerate(TICKERS) if round(d_units * w[i]) != 0]


class EtfOnly:
    """What most teams ship first: fixed threshold, ETF leg only, market orders."""
    name = "1 ETF-only threshold"

    def __init__(self, entry=0.0030, exit_=0.0005, notional=2_000_000):
        self.entry, self.exit, self.notional = entry, exit_, notional

    def on_tick(self, v):
        prem = v.etf_mid / v.nav - 1
        pos = v.etf_pos
        if prem > self.entry and pos >= 0:
            tgt = -round(self.notional / v.etf_mid)
        elif prem < -self.entry and pos <= 0:
            tgt = round(self.notional / v.etf_mid)
        elif abs(prem) < self.exit:
            tgt = 0
        else:
            tgt = pos
        return [Market("ETF", tgt - pos)] if tgt != pos else []


class Hedged(EtfOnly):
    """Same signal, but hedged with the basket: PnL from convergence, not direction."""
    name = "2 hedged threshold"

    def __init__(self, entry=0.0035, exit_=0.0005, notional=2_000_000):
        super().__init__(entry, exit_, notional)

    def on_tick(self, v):
        orders = super().on_tick(v)
        for o in list(orders):
            orders += basket_orders(o.qty)
        return orders


class ZScore:
    """
    Adaptive hedged arb:
      * premium measured against its own rolling mean and stdev (z-score)
      * entry must ALSO clear the hedged round-trip cost with margin
      * signal must persist 2 ticks (filters stale quotes / one-tick blips)
      * no new entries for a few ticks after news (spreads are wide, moves are noisy)
      * size scales with conviction; exits when z is back near 0
    """
    name = "3 z-score hedged"

    def __init__(self, window=300, z_in=2.0, z_out=0.3, notional=2_000_000, cost=0.0018):
        self.w, self.z_in, self.z_out, self.notional, self.cost = window, z_in, z_out, notional, cost
        self.hist, self.streak, self.cooldown = [], 0, 0

    def on_tick(self, v):
        prem = v.etf_mid / v.nav - 1
        self.hist.append(prem)
        if v.news:
            self.cooldown = 10
        self.cooldown = max(0, self.cooldown - 1)
        if len(self.hist) < 50:
            return []
        h = np.array(self.hist[-self.w:])
        mu, sd = h.mean(), h.std() + 1e-9
        z = (prem - mu) / sd
        dev = abs(prem - mu)
        self.streak = self.streak + 1 if (abs(z) > self.z_in and dev > self.cost) else 0

        pos = v.etf_pos
        tgt = pos
        if self.streak >= 2 and self.cooldown == 0:
            clips = 2 if abs(z) > 3 else 1
            want = -np.sign(z) * clips * round(self.notional / v.etf_mid)
            if np.sign(want) != np.sign(pos) or abs(want) > abs(pos):
                tgt = want
        elif abs(z) < self.z_out:
            tgt = 0
        d = tgt - pos
        return ([Market("ETF", d)] + basket_orders(d)) if d else []

    @staticmethod
    def half_life(series):
        """AR(1) fit: x_t = phi x_{t-1} + e  ->  half-life = ln(0.5)/ln(phi) ticks."""
        x = np.asarray(series)
        phi = np.dot(x[1:], x[:-1]) / np.dot(x[:-1], x[:-1])
        return math.log(0.5) / math.log(phi) if 0 < phi < 1 else float("inf")


class MarketMaker:
    """
    Quote the ETF around fair value (NAV + typical premium) and lean on inventory:
      reservation price  r = fair x (1 - k x inventory)       (Avellaneda-Stoikov intuition)
      half-spread        h = max(base, m x recent tick volatility of the ETF)
    Your fills are partial and adverse-selected: informed traders hit you just before
    price moves through your quote. Two defences: quote wider than one tick of
    volatility, and hedge inventory in the basket quickly (small band).
    Step back from the market for a few ticks after news.
    """
    name = "4 market maker"

    def __init__(self, clip=1500, base_hs=0.0004, vol_mult=2.0, skew_per_unit=1.2e-7, band=750):
        self.clip, self.base_hs, self.vol_mult = clip, base_hs, vol_mult
        self.skew, self.band = skew_per_unit, band
        self.hist, self.mids, self.hedged_units, self.pause = [], [], 0.0, 0

    def on_tick(self, v):
        nav = v.nav
        self.hist.append(v.etf_mid / nav - 1)
        self.mids.append(v.etf_mid)
        h = np.array(self.hist[-200:])
        fair = nav * (1 + (h.mean() if len(h) > 20 else 0))
        m = np.array(self.mids[-100:])
        tick_vol = np.diff(np.log(m)).std() if len(m) > 20 else 0.001
        if v.news:
            self.pause = 5
        if self.pause:
            self.pause -= 1
            return [CancelQuotes("ETF")]
        if v.ticks_left < 40:
            return [CancelQuotes("ETF")]
        inv = v.etf_pos + self.hedged_units                   # ETF units not yet hedged
        r = fair * (1 - self.skew * inv)                      # lean against inventory
        hs = max(self.base_hs, self.vol_mult * tick_vol)
        bid_qty = self.clip if inv < 2 * self.clip else 0     # hard inventory cap
        ask_qty = self.clip if inv > -2 * self.clip else 0
        orders = [Quote("ETF", r * (1 - hs), bid_qty, r * (1 + hs), ask_qty)]
        if abs(inv) > self.band:                              # hedge it all in the basket
            orders += basket_orders(inv)
            self.hedged_units -= inv
        return orders


# =============================================================== YOUR strategy
class MyStrategy:
    """Edit me. Run with --mine. Start from ZScore or MarketMaker and change ONE thing at a time."""
    name = "5 mine"

    def __init__(self):
        self.inner = ZScore()

    def on_tick(self, v):
        return self.inner.on_tick(v)


# =============================================================== reporting
def fmt(x):
    return f"{'-' if x < 0 else ''}${abs(x):,.0f}"


def compare(strats, seeds):
    print(f"{'strategy':<22}{'PnL (avg)':>12}{'worst seed':>12}{'Sharpe':>8}{'max DD':>11}"
          f"{'fees':>10}{'spread':>10}{'avg |net|':>12}{'beta PnL':>11}")
    for make in strats:
        rs = [run(make(), seed=s) for s in seeds]
        a = lambda k: float(np.mean([r[k] for r in rs]))
        worst = min(r["pnl"] for r in rs)
        print(f"{make().name:<22}{fmt(a('pnl')):>12}{fmt(worst):>12}{a('sharpe'):>8.2f}{fmt(a('maxdd')):>11}"
              f"{fmt(a('fees')):>10}{fmt(a('spread')):>10}{fmt(a('avg_net')):>12}{fmt(a('beta_pnl')):>11}")


DRILL = """
15-MINUTE DRILL  (no AI, no peeking; start a timer)
 1. Write nav(stock_mid, shares_per_unit) and premium(etf_mid, nav).            [2 min]
 2. Write on_tick for a hedged threshold strategy using Market orders.          [6 min]
    - enter at |premium| > entry, exit at |premium| < exit, flatten near the end
    - hedge with whole-share basket orders
 3. Add: persistence filter (2 ticks), news cooldown, position cap.             [4 min]
 4. Paste it into MyStrategy, run `python quant_edge_lab.py --mine`.            [3 min]
Pass mark: runs first time, positive PnL on the average AND the worst seed.
Then: explain out loud where the PnL came from (convergence vs beta) in 30 s.
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--mine", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--drill", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="threshold and spread sweeps")
    a = ap.parse_args()
    if a.drill:
        print(DRILL)
        raise SystemExit
    md = Market_Data(seed=0)
    print(f"Premium: sd {md.prem.std():.3%}, half-life {ZScore.half_life(md.prem):.1f} ticks, "
          f"news events {md.news.sum()}, fee {FEE:.4%}/fill\n")
    strats = [EtfOnly, Hedged, ZScore, MarketMaker] + ([MyStrategy] if a.mine else [])
    compare(strats, range(1, a.seeds + 1))
    print("\nHow to read this: 'beta PnL' is the part explained by the market moving while you held net"
          "\nexposure. A real arbitrage keeps it near zero. 'spread' is what you paid to cross the book.")
    if a.sweep:
        seeds = range(1, a.seeds + 1)
        avg = lambda make: (np.mean([run(make(), seed=s)["pnl"] for s in seeds]),
                            min(run(make(), seed=s)["pnl"] for s in seeds))
        print("\nHedged threshold: entry sweep (too tight = fees and spread eat you)")
        for e in (0.0010, 0.0020, 0.0030, 0.0035, 0.0045, 0.0060):
            m, w = avg(lambda: Hedged(entry=e))
            print(f"  entry {e:.2%}   avg {fmt(m):>12}   worst {fmt(w):>12}")
        print("\nMarket maker: half-spread as a multiple of tick volatility (too tight = picked off)")
        for vm in (0.6, 1.0, 1.6, 2.0, 2.5, 3.5):
            m, w = avg(lambda: MarketMaker(vol_mult=vm))
            print(f"  {vm:>3}x vol   avg {fmt(m):>12}   worst {fmt(w):>12}")
    if a.plot:
        import matplotlib.pyplot as plt
        for make in strats:
            plt.plot(run(make(), seed=1, record=True)["path"], label=make().name)
        plt.legend(); plt.xlabel("tick"); plt.ylabel("PnL ($)"); plt.title("Seed 1")
        plt.savefig("pnl.png", dpi=130, bbox_inches="tight")
        print("saved pnl.png")
