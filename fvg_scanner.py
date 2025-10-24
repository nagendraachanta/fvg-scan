import yfinance as yf
import pandas as pd
import argparse
import os

TRACKER_FILE = "fvg_tracker.csv"  # persistent file with only OPEN (unbreached) FVGs

COLUMNS = ["Ticker", "Type", "GapLow", "GapHigh", "Start", "End", "FID"]

# ---------- Data helpers ----------

def flatten_ohlc(df, prefer_ticker=None):
    """Return DataFrame with columns Open, High, Low, Close, Volume from either flat or MultiIndex."""
    if not isinstance(df.columns, pd.MultiIndex):
        cols = {c.lower(): c for c in df.columns}
        need = ["open", "high", "low", "close", "volume"]
        missing = [n for n in need if n not in cols]
        if missing:
            raise KeyError(f"Missing columns {missing} in {list(df.columns)}")
        out = df[[cols["open"], cols["high"], cols["low"], cols["close"], cols["volume"]]].copy()
        out.columns = ["Open", "High", "Low", "Close", "Volume"]
        return out

    # MultiIndex case (e.g., levels ['Price','Ticker'] or ['Ticker','Price'])
    lvl0 = [str(x).lower() for x in df.columns.get_level_values(0)]
    lvl1 = [str(x).lower() for x in df.columns.get_level_values(1)]
    ohlc = {"open", "high", "low", "close", "volume"}

    if ohlc.issubset(set(lvl0)):
        price_level = 0
    elif ohlc.issubset(set(lvl1)):
        price_level = 1
    else:
        raise KeyError(f"Could not find OHLC level in columns: {df.columns}")

    out = {}
    for name in ["Open", "High", "Low", "Close", "Volume"]:
        sub = df.xs(name, level=price_level, axis=1, drop_level=True)
        if isinstance(sub, pd.Series):
            out[name] = sub
        else:
            if prefer_ticker and prefer_ticker in sub.columns:
                out[name] = sub[prefer_ticker]
            else:
                # single ticker download usually yields exactly one column
                out[name] = sub.iloc[:, 0]
    return pd.DataFrame(out, index=df.index)

def make_fid(ticker, fvg_type, start_ts, end_ts, gap_low, gap_high):
    s = pd.to_datetime(start_ts).date()
    e = pd.to_datetime(end_ts).date()
    return f"{ticker}|{fvg_type}|{s}|{e}|{gap_low:.6f}|{gap_high:.6f}"

# ---------- FVG logic ----------

def detect_fvg(df, ticker):
    """ICT 3-candle FVG detection."""
    gaps = []
    for i in range(2, len(df)):
        high1 = float(df["High"].iloc[i - 2])
        low1  = float(df["Low"].iloc[i - 2])
        high3 = float(df["High"].iloc[i])
        low3  = float(df["Low"].iloc[i])

        # Bearish FVG: C3 high < C1 low
        if high3 < low1:
            start_ts = df.index[i - 2]
            end_ts   = df.index[i]
            gap_low  = high3
            gap_high = low1
            gaps.append({
                "Ticker": ticker, "Type": "bearish",
                "GapLow": gap_low, "GapHigh": gap_high,
                "Start": start_ts, "End": end_ts,
                "FID": make_fid(ticker, "bearish", start_ts, end_ts, gap_low, gap_high),
            })

        # Bullish FVG: C3 low > C1 high
        if low3 > high1:
            start_ts = df.index[i - 2]
            end_ts   = df.index[i]
            gap_low  = high1
            gap_high = low3
            gaps.append({
                "Ticker": ticker, "Type": "bullish",
                "GapLow": gap_low, "GapHigh": gap_high,
                "Start": start_ts, "End": end_ts,
                "FID": make_fid(ticker, "bullish", start_ts, end_ts, gap_low, gap_high),
            })
    return gaps

def first_breach_date(df, end_ts, fvg_type, gap_low, gap_high):
    """
    Scan CLOSES after the FVG End bar and return first breach timestamp (or None).
    STRICT rules:
      - bearish breach: close > gap_high
      - bullish breach: close < gap_low
    """
    end_ts = pd.to_datetime(end_ts)
    try:
        end_idx = df.index.get_loc(end_ts)
    except KeyError:
        end_idx = df.index.get_indexer([end_ts], method="nearest")[0]

    closes = df["Close"].iloc[end_idx + 1:]
    if closes.empty:
        return None

    if fvg_type == "bearish":
        hits = closes[closes > gap_high]
    else:
        hits = closes[closes < gap_low]

    return None if hits.empty else hits.index[0]

# ---------- Tracker I/O ----------

def load_tracker():
    if os.path.exists(TRACKER_FILE):
        try:
            df = pd.read_csv(TRACKER_FILE, parse_dates=["Start", "End"])
            for c in COLUMNS:
                if c not in df.columns:
                    df[c] = pd.Series(dtype="object")
            df = df[COLUMNS]
            # rebuild FID if missing
            miss = df["FID"].isna() | (df["FID"] == "")
            if miss.any():
                df.loc[miss, "FID"] = df.loc[miss].apply(
                    lambda r: make_fid(r["Ticker"], r["Type"], r["Start"], r["End"],
                                       float(r["GapLow"]), float(r["GapHigh"])),
                    axis=1,
                )
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=COLUMNS)

def save_tracker(df):
    if not df.empty:
        df = df[COLUMNS].sort_values(["Ticker", "Start", "End"]).reset_index(drop=True)
    df.to_csv(TRACKER_FILE, index=False)

# ---------- Main ----------

def main():
    p = argparse.ArgumentParser(description="ICT 3-Candle FVG Tracker (strict breach, clears on breach)")
    p.add_argument("--tickers", type=str, default="watchlist.csv", help="CSV: single column of tickers, no header")
    p.add_argument("--timeframe", type=str, choices=["1d", "1wk"], default="1d")
    p.add_argument("--period", type=str, default="2y", help="yfinance period, e.g. 1y, 2y, 5y")
    args = p.parse_args()

    tickers = pd.read_csv(args.tickers, header=None).iloc[:, 0].astype(str).str.strip()
    tickers = [t for t in tickers if t]
    print(f"Tickers loaded: {tickers}\n")

    tracker = load_tracker()
    open_rows_all = []   # rebuilt tracker with only open FVGs
    breaches_today = []  # breaches that occurred on the most recent bar
    all_breaches = []    # every breach found this run (with breach date)

    for idx, ticker in enumerate(tickers, 1):
        print(f"[{idx}/{len(tickers)}] Scanning {ticker}...")
        try:
            raw = yf.download(ticker, period=args.period, interval=args.timeframe,
                              auto_adjust=True, progress=False)
            if len(raw) < 3:
                print(f"  Skipped {ticker}: insufficient bars.")
                continue

            df = flatten_ohlc(raw, prefer_ticker=ticker)
            last_ts = df.index[-1]
            last_date = last_ts.date()

            # detect new FVGs
            detected = detect_fvg(df, ticker)

            # dedupe by FID
            fid_set = set(tracker["FID"].astype(str)) if not tracker.empty else set()
            to_add = [g for g in detected if g["FID"] not in fid_set]
            if to_add:
                tracker = pd.concat([tracker, pd.DataFrame(to_add)], ignore_index=True)
                fid_set.update([g["FID"] for g in to_add])

            # only this ticker's FVGs for breach scan
            trows = tracker[tracker["Ticker"] == ticker]
            if trows.empty:
                continue

            for _, r in trows.iterrows():
                gap_low  = float(r["GapLow"])
                gap_high = float(r["GapHigh"])
                ftype    = r["Type"]
                end_ts   = r["End"]

                b_ts = first_breach_date(df, end_ts, ftype, gap_low, gap_high)
                if b_ts is None:
                    # still open
                    open_rows_all.append({
                        "Ticker": ticker, "Type": ftype,
                        "GapLow": gap_low, "GapHigh": gap_high,
                        "Start": r["Start"], "End": end_ts, "FID": r["FID"]
                    })
                else:
                    # breached
                    entry = {
                        "Ticker": ticker, "Type": ftype,
                        "GapLow": gap_low, "GapHigh": gap_high,
                        "Start": r["Start"], "End": end_ts,
                        "BreachTS": b_ts, "BreachDate": pd.to_datetime(b_ts).date(),
                        "FID": r["FID"]
                    }
                    all_breaches.append(entry)
                    if pd.to_datetime(b_ts).date() == last_date:
                        breaches_today.append(entry)

        except Exception as e:
            print(f"  Error scanning {ticker}: {e}")
            continue

    # save tracker with only OPEN FVGs
    tracker = pd.DataFrame(open_rows_all, columns=COLUMNS)
    save_tracker(tracker)

    # write latest-bar breaches (only last breach per ticker, sorted by Type then Ticker)
    if breaches_today:
        df_today = pd.DataFrame(breaches_today)
        df_today = (
            df_today.sort_values(["Ticker", "BreachTS"])
                    .groupby("Ticker", as_index=False)
                    .tail(1)
                    .sort_values(["Type", "Ticker"])
        )
        df_today.to_csv("breaches_today.csv", index=False)
        print(f"\nFound {len(df_today)} breach(es) on the latest bar. Saved to breaches_today.csv")
    else:
        pd.DataFrame(columns=[
            "Ticker","Type","GapLow","GapHigh","Start","End","BreachTS","BreachDate","FID"
        ]).to_csv("breaches_today.csv", index=False)
        print("\nNo breaches on the latest bar.")

    # full breach log for the run (all breaches, chronological)
    #if all_breaches:
     #   pd.DataFrame(all_breaches).to_csv("breaches_log.csv", index=False)
      #  print(f"Logged {len(all_breaches)} total breach(es) this run to breaches_log.csv")

if __name__ == "__main__":
    main()
