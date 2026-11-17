# SPDX-License-Identifier: MIT
"""Dip-ensemble BTC trading bot for make.fun agent-fun/bars-replay/v1.

Strategy (derived from the competition dataset, integer math only):
  Entries: an OR of multi-scale dip legs, all measured on closes versus the
  trailing maximum of *previous* bars (current bar excluded):
    A: 60-bar dip <= -120 bps
    B: 120-bar dip <= -130 bps
    C: 240-bar dip <= -160 bps while the close is the trailing 20-bar low
    D: early window only (first 60 bars): 10-bar dip <= -35 bps while the
       close is the trailing 10-bar low
  Late phase (bar index >= 1000): only leg C with a deeper -170 bps dip.
  A cooldown after each exit delays re-entry: 45 bars after a trailing/stop
  exit, 59 bars after a take-profit exit (counted from the sell fill bar).

  Exits (signal close = close of the bar that triggered the entry):
    * take profit at +200 bps versus the signal close;
    * delayed trailing stop: inactive for the first 120 bars of a trade,
      then a tight 20 bps trail while the peak gain is below +130 bps, and
      a wide 80 bps trail once the peak gain reaches +130 bps (lets big
      runners ride through chop, cuts small tops quickly);
    * disaster stop at -150 bps versus the signal close;
    * hold cap of 480 bars.
  All-in sizing, one position at a time, long only. Deterministic: pure
  integer arithmetic over past bars only, no clock, network or randomness.
"""

import json
import sys

SYMBOL_FALLBACK = "CBBTC"
MAX_SLIPPAGE_BPS = 10000

# Entry thresholds (basis points, dip = (close - trailing max) * 10000 // max).
DIP_60 = -120
DIP_120 = -130
DIP_240 = -160
DIP_240_LATE = -170
DIP_10_EARLY = -35
EARLY_BARS = 60
LATE_BAR = 1000
MIN_WINDOW = 20
MIN_WINDOW_EARLY = 10

# Cooldowns in bars, counted from the sell fill bar.
COOLDOWN_AFTER_TRAIL = 45
COOLDOWN_AFTER_TP = 59

# Exit parameters (basis points versus the signal close / running peak).
TAKE_PROFIT_BPS = 200
TRAIL_DELAY_BARS = 120
TRAIL_ARM_BPS = 60
TRAIL_SWITCH_BPS = 130
TRAIL_TIGHT_BPS = 20
TRAIL_WIDE_BPS = 80
STOP_BPS = 150
HOLD_CAP_BARS = 480


def dip_vs_prev_max(closes, idx, window):
    """Dip of closes[idx] versus the max of the previous `window` closes.

    Returns 0 when there is no previous bar. Floor division matches the
    integer convention used throughout (negative dips round down).
    """
    if idx <= 0:
        return 0
    lo = idx - window
    if lo < 0:
        lo = 0
    prev_max = closes[lo]
    for j in range(lo + 1, idx):
        if closes[j] > prev_max:
            prev_max = closes[j]
    if prev_max <= 0:
        return 0
    return (closes[idx] - prev_max) * 10000 // prev_max


def is_trailing_low(closes, idx, window):
    """True when closes[idx] is the minimum of the last `window` closes."""
    lo = idx - window + 1
    if lo < 0:
        lo = 0
    c = closes[idx]
    for j in range(lo, idx):
        if closes[j] < c:
            return False
    return True


class Bot:
    def __init__(self):
        self.symbol = SYMBOL_FALLBACK
        self.closes = []
        self.cash = 0
        self.qty = 0
        # Position state, driven by authoritative fills/account.
        self.sig_close = None
        self.peak = None
        self.fill_bar = None
        self.last_flat_bar = -10 ** 9
        self.last_exit_was_tp = False
        self.pending_buy_sig = None
        self.pending_sell_tp = False
        self.order_no = 0

    def handle_hello(self, msg):
        symbols = msg.get("symbols") or []
        if symbols:
            sym = symbols[0].get("symbol")
            if sym:
                self.symbol = sym
        capital = msg.get("capital") or {}
        try:
            self.cash = int(capital.get("quote_units", "0"))
        except (TypeError, ValueError):
            self.cash = 0

    def handle_batch(self, msg):
        orders = []
        # 1. Ingest our bar (a batch may carry no bar for our symbol).
        bar_close = None
        for bar in msg.get("bars") or []:
            if bar.get("symbol") == self.symbol:
                try:
                    bar_close = int(bar["c"])
                except (KeyError, TypeError, ValueError):
                    bar_close = None
                break
        if bar_close is not None:
            self.closes.append(bar_close)
        bar_idx = len(self.closes) - 1

        # 2. Apply fills (authoritative for position transitions).
        for fill in msg.get("fills") or []:
            if fill.get("symbol") != self.symbol:
                continue
            if fill.get("status") != "filled":
                continue
            side = fill.get("side")
            if side == "buy":
                # Fall back to the current bar close if no signal was
                # captured (e.g. an order acknowledged out of order), so a
                # filled position always has a reference price for exits.
                self.sig_close = (
                    self.pending_buy_sig if self.pending_buy_sig is not None else bar_close
                )
                self.fill_bar = bar_idx
                self.peak = bar_close
                self.pending_buy_sig = None
            elif side == "sell":
                self.last_flat_bar = bar_idx
                self.last_exit_was_tp = self.pending_sell_tp
                self.pending_sell_tp = False
                self.sig_close = None
                self.peak = None
                self.fill_bar = None

        # 3. Sync cash/qty from the authoritative account snapshot.
        account = msg.get("account") or {}
        try:
            self.cash = int(account.get("cash_quote_units", "0"))
        except (TypeError, ValueError):
            pass
        qty = 0
        for pos in account.get("positions") or []:
            if pos.get("symbol") == self.symbol:
                try:
                    qty += int(pos.get("qty", "0"))
                except (TypeError, ValueError):
                    pass
        self.qty = qty

        # Reconcile position state with the authoritative account snapshot:
        # if a fill report was ever missed, re-derive the exit reference
        # from the current bar (long) or clear it (flat) instead of trading
        # on stale state. Never triggers when fills arrive normally.
        if self.qty > 0 and self.sig_close is None and bar_close is not None:
            self.sig_close = bar_close
            self.peak = bar_close
            self.fill_bar = bar_idx
        elif self.qty <= 0 and self.sig_close is not None:
            self.last_flat_bar = bar_idx
            self.sig_close = None
            self.peak = None
            self.fill_bar = None

        # 4. Decide (only on batches that carried our bar).
        if bar_close is not None:
            if self.qty <= 0:
                orders = self.maybe_enter(bar_idx, bar_close)
            elif self.sig_close is not None:
                orders = self.maybe_exit(bar_idx, bar_close)
        return orders

    def maybe_enter(self, idx, close):
        cooldown = COOLDOWN_AFTER_TP if self.last_exit_was_tp else COOLDOWN_AFTER_TRAIL
        if idx - self.last_flat_bar < cooldown:
            return []
        fire = False
        if idx >= LATE_BAR:
            if dip_vs_prev_max(self.closes, idx, 240) <= DIP_240_LATE and is_trailing_low(
                self.closes, idx, MIN_WINDOW
            ):
                fire = True
        else:
            if dip_vs_prev_max(self.closes, idx, 60) <= DIP_60:
                fire = True
            elif dip_vs_prev_max(self.closes, idx, 120) <= DIP_120:
                fire = True
            elif dip_vs_prev_max(self.closes, idx, 240) <= DIP_240 and is_trailing_low(
                self.closes, idx, MIN_WINDOW
            ):
                fire = True
            elif (
                idx < EARLY_BARS
                and dip_vs_prev_max(self.closes, idx, 10) <= DIP_10_EARLY
                and is_trailing_low(self.closes, idx, MIN_WINDOW_EARLY)
            ):
                fire = True
        if not fire:
            return []
        if self.cash <= 0:
            return []
        self.order_no += 1
        self.pending_buy_sig = close
        return [
            {
                "side": "buy",
                "client_id": "b%d" % self.order_no,
                "symbol": self.symbol,
                "notional_quote_units": str(self.cash),
                "max_slippage_bps": MAX_SLIPPAGE_BPS,
            }
        ]

    def maybe_exit(self, idx, close):
        sig = self.sig_close
        if self.peak is None or close > self.peak:
            self.peak = close
        peak = self.peak
        held = idx - self.fill_bar if self.fill_bar is not None else 0
        was_tp = False
        fire = False
        # Take profit versus the signal close.
        if close * 10000 >= sig * (10000 + TAKE_PROFIT_BPS):
            fire = True
            was_tp = True
        # Delayed trailing stop.
        elif held >= TRAIL_DELAY_BARS:
            peak_gain = (peak - sig) * 10000 // sig if sig > 0 else 0
            if peak_gain < TRAIL_SWITCH_BPS:
                if peak_gain >= TRAIL_ARM_BPS and close * 10000 <= peak * (
                    10000 - TRAIL_TIGHT_BPS
                ):
                    fire = True
            elif close * 10000 <= peak * (10000 - TRAIL_WIDE_BPS):
                fire = True
        # Disaster stop.
        if not fire and close * 10000 <= sig * (10000 - STOP_BPS):
            fire = True
        # Hold cap.
        if not fire and held >= HOLD_CAP_BARS:
            fire = True
        if not fire:
            return []
        self.order_no += 1
        self.pending_sell_tp = was_tp
        return [
            {
                "side": "sell",
                "client_id": "s%d" % self.order_no,
                "symbol": self.symbol,
                "qty": str(self.qty),
                "max_slippage_bps": MAX_SLIPPAGE_BPS,
            }
        ]


def main():
    bot = Bot()
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        mtype = msg.get("type")
        if mtype == "hello":
            bot.handle_hello(msg)
        elif mtype == "batch":
            try:
                orders = bot.handle_batch(msg)
            except Exception:
                orders = []
            reply = {"type": "reply", "seq": msg.get("seq", 0), "orders": orders}
            out.write(json.dumps(reply) + "\n")
            out.flush()
        elif mtype == "done":
            break


if __name__ == "__main__":
    main()
