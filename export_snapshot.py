"""Export snapshot data from PostgreSQL for the Streamlit app.

Connects to the local fundamental_analysis database and generates all
snapshot files in data/.  Replicates the logic from the InvestingProject
dashboard backend services.

Usage:
    python export_snapshot.py                              # uses default run_id
    python export_snapshot.py --run-id bt_20260218_0022_sparse_qp_bl1.0_t40_mp0.002
"""
import argparse
import json
import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://fa_user:fa_secure_password_2024@localhost:5432/fundamental_analysis"
DEFAULT_RUN_ID = "bt_20260218_0022_sparse_qp_bl1.0_t40_mp0.002"
DATA_DIR = Path(__file__).resolve().parent / "data"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso_to_friday(iso_year: int, iso_week: int):
    from datetime import date
    return date.fromisocalendar(iso_year, iso_week, 5)


def _load_benchmark(start_date, end_date) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT iso_year, iso_week, weekly_return
            FROM market_returns_weekly
            WHERE symbol = '^GSPC'
            ORDER BY iso_year, iso_week
        """), conn)
    if df.empty:
        return pd.DataFrame(columns=["date", "benchmark_return"])
    df["date"] = df.apply(lambda r: _iso_to_friday(int(r["iso_year"]), int(r["iso_week"])), axis=1)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"weekly_return": "benchmark_return"})
    df["benchmark_return"] = pd.to_numeric(df["benchmark_return"], errors="coerce")
    df = df.dropna(subset=["benchmark_return"])
    df = df[(df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))]
    df = df.drop_duplicates(subset=["date"], keep="last")
    return df[["date", "benchmark_return"]].sort_values("date").reset_index(drop=True)


def _load_ohlcv(symbols, start_date, end_date) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT symbol, date, adj_close
            FROM ohlcv_daily
            WHERE symbol = ANY(:symbols)
              AND date BETWEEN :start AND :end
              AND adj_close IS NOT NULL
            ORDER BY symbol, date
        """), conn, params={"symbols": symbols, "start": start_date, "end": end_date})
    df["date"] = pd.to_datetime(df["date"])
    df["adj_close"] = df["adj_close"].astype(float)
    return df


def _weekly_prices(ohlcv: pd.DataFrame) -> pd.DataFrame:
    if ohlcv.empty:
        return pd.DataFrame()
    ohlcv = ohlcv.copy()
    ohlcv["week_friday"] = ohlcv["date"] + pd.to_timedelta(4 - ohlcv["date"].dt.weekday, unit="D")
    weekly = ohlcv.sort_values("date").groupby(["symbol", "week_friday"]).last().reset_index()
    return weekly.pivot(index="week_friday", columns="symbol", values="adj_close").sort_index()


def _get_coverage_start_date() -> str | None:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT as_of_date,
                   100.0 * COUNT(implied_price_dcf_mc_median) / COUNT(*) AS dcf_mc_pct,
                   100.0 * COUNT(implied_price_ev_ebitda)     / COUNT(*) AS ev_ebitda_pct,
                   100.0 * COUNT(implied_price_pe)            / COUNT(*) AS pe_pct,
                   100.0 * COUNT(implied_price_ddm)           / COUNT(*) AS ddm_pct
            FROM valuation_implied_prices
            GROUP BY as_of_date
            ORDER BY as_of_date
        """))
        for row in rows:
            as_of_date, dcf_pct, ev_pct, pe_pct, ddm_pct = row
            if dcf_pct >= 30 and ev_pct >= 30 and pe_pct >= 30 and ddm_pct >= 30:
                log.info(f"Coverage starts at {as_of_date}")
                return str(as_of_date)
    return None


# ---------------------------------------------------------------------------
# KPI computation
# ---------------------------------------------------------------------------

def _compute_kpis(merged: pd.DataFrame, rebalance_dates, weight_lookup, rf: float) -> dict:
    port_ret = merged["portfolio_return"].values.astype(float)
    n_weeks = len(port_ret)
    if n_weeks == 0:
        return _empty_kpis()
    years = n_weeks / 52.0

    cum_port = np.prod(1 + port_ret)
    total_return = cum_port - 1
    cagr = cum_port ** (1 / years) - 1 if years > 0 else 0.0
    weekly_vol = float(np.std(port_ret, ddof=1)) if n_weeks > 1 else 0.0
    ann_vol = weekly_vol * np.sqrt(52)
    rf_weekly = (1 + rf) ** (1 / 52.0) - 1
    mean_weekly = float(np.mean(port_ret))
    sharpe = ((mean_weekly - rf_weekly) / weekly_vol) * np.sqrt(52) if weekly_vol > 0 else 0.0

    downside = port_ret[port_ret < 0]
    downside_vol = np.std(downside, ddof=1) * np.sqrt(52) if len(downside) > 1 else ann_vol
    sortino = (cagr - rf) / downside_vol if downside_vol > 0 else 0.0

    cum_curve = np.cumprod(1 + port_ret)
    running_max = np.maximum.accumulate(cum_curve)
    dd_series = (cum_curve / running_max) - 1
    max_dd = float(np.min(dd_series))
    current_dd = float(dd_series[-1])

    max_dd_dur = cur_dur = 0
    for dd in dd_series:
        if dd < 0:
            cur_dur += 1
            max_dd_dur = max(max_dd_dur, cur_dur)
        else:
            cur_dur = 0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    bench_aligned = merged.dropna(subset=["benchmark_return"]).copy()
    bench_ret = bench_aligned["benchmark_return"].values.astype(float)
    port_ret_for_bench = bench_aligned["portfolio_return"].values.astype(float)
    bench_years = len(bench_ret) / 52.0 if len(bench_ret) > 0 else 0.0
    if len(bench_ret) > 1 and np.std(bench_ret, ddof=1) > 0 and bench_years > 0:
        cov_pb = np.cov(port_ret_for_bench, bench_ret)[0, 1]
        var_b = np.var(bench_ret, ddof=1)
        beta = cov_pb / var_b if var_b > 0 else 1.0
        bench_cum = np.prod(1 + bench_ret)
        bench_cagr = bench_cum ** (1 / bench_years) - 1
        port_cum_for_bench = np.prod(1 + port_ret_for_bench)
        port_cagr_for_bench = port_cum_for_bench ** (1 / bench_years) - 1
        alpha = port_cagr_for_bench - rf - beta * (bench_cagr - rf)
    else:
        beta, alpha = 1.0, 0.0

    if len(bench_ret) > 1:
        excess = port_ret_for_bench - bench_ret
        te = np.std(excess, ddof=1) * np.sqrt(52)
        ir = (np.mean(excess) * 52) / te if te > 0 else 0.0
    else:
        ir = 0.0

    monthly_df = merged.copy()
    monthly_df["month"] = monthly_df["date"].dt.to_period("M")

    def _compound(s):
        s = s.dropna()
        return (1 + s).prod() - 1 if not s.empty else np.nan

    monthly = monthly_df.groupby("month").apply(
        lambda g: pd.Series({
            "portfolio_return": (1 + g["portfolio_return"]).prod() - 1,
            "benchmark_return": _compound(g["benchmark_return"]),
        }),
        include_groups=False,
    ).reset_index()
    monthly["month"] = monthly["month"].dt.to_timestamp()

    m_ret = monthly["portfolio_return"].values.astype(float)
    win_rate = float(np.mean(m_ret > 0)) if len(m_ret) > 0 else 0.0
    avg_monthly = float(np.mean(m_ret)) if len(m_ret) > 0 else 0.0
    best_month = float(np.max(m_ret)) if len(m_ret) > 0 else 0.0
    worst_month = float(np.min(m_ret)) if len(m_ret) > 0 else 0.0

    merge_start = merged["date"].min()
    filtered_rebals = [d for d in rebalance_dates if d >= merge_start]
    turnovers = []
    prev_w = None
    for rd in filtered_rebals:
        w = weight_lookup[rd]
        if prev_w is not None:
            all_syms = prev_w.index.union(w.index)
            wo = prev_w.reindex(all_syms, fill_value=0)
            wn = w.reindex(all_syms, fill_value=0)
            turnovers.append(float(np.abs(wn - wo).sum()))
        prev_w = w
    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0

    return {
        "cagr": round(cagr, 6),
        "annualized_vol": round(ann_vol, 6),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_dd, 6),
        "max_dd_duration_weeks": max_dd_dur,
        "calmar": round(calmar, 4),
        "beta": round(beta, 4),
        "alpha": round(alpha, 6),
        "information_ratio": round(ir, 4),
        "win_rate": round(win_rate, 4),
        "avg_monthly_return": round(avg_monthly, 6),
        "best_month": round(best_month, 6),
        "worst_month": round(worst_month, 6),
        "current_drawdown": round(current_dd, 6),
        "avg_turnover": round(avg_turnover, 4),
        "total_return": round(total_return, 6),
        "n_weeks": n_weeks,
        "n_months": len(m_ret),
        "start_date": str(merged["date"].min().date()),
        "end_date": str(merged["date"].max().date()),
    }


def _empty_kpis():
    return {
        "cagr": 0, "annualized_vol": 0, "sharpe": 0, "sortino": 0,
        "max_drawdown": 0, "max_dd_duration_weeks": 0, "calmar": 0,
        "beta": 0, "alpha": 0, "information_ratio": 0, "win_rate": 0,
        "avg_monthly_return": 0, "best_month": 0, "worst_month": 0,
        "current_drawdown": 0, "avg_turnover": 0, "total_return": 0,
        "n_weeks": 0, "n_months": 0, "start_date": "", "end_date": "",
    }


# ---------------------------------------------------------------------------
# 1. Performance snapshot
# ---------------------------------------------------------------------------

def export_performance(run_id: str) -> dict:
    log.info("Exporting performance snapshot...")
    rf = 0.02

    with engine.connect() as conn:
        snapshots = pd.read_sql(text("""
            SELECT symbol, as_of_date, weight
            FROM portfolio_snapshots
            WHERE run_id = :run_id
            ORDER BY as_of_date, symbol
        """), conn, params={"run_id": run_id})

    if snapshots.empty:
        raise RuntimeError(f"No snapshots found for run_id={run_id}")

    snapshots["as_of_date"] = pd.to_datetime(snapshots["as_of_date"])
    snapshots["weight"] = snapshots["weight"].astype(float)

    rebalance_dates = sorted(snapshots["as_of_date"].unique())
    start_date = rebalance_dates[0]
    end_date = rebalance_dates[-1]
    all_symbols = sorted(snapshots["symbol"].unique().tolist())
    log.info(f"  {len(rebalance_dates)} rebalance dates, {len(all_symbols)} unique symbols")

    price_start = start_date - timedelta(days=10)
    price_end = end_date + pd.DateOffset(months=2)
    ohlcv = _load_ohlcv(all_symbols, price_start, price_end)
    wp = _weekly_prices(ohlcv)

    if wp.empty or len(wp) < 2:
        raise RuntimeError("Insufficient price data")

    weekly_returns = wp.pct_change(fill_method=None).iloc[1:]
    benchmark = _load_benchmark(price_start, price_end)

    weight_lookup = {}
    for d in rebalance_dates:
        mask = snapshots["as_of_date"] == d
        w = snapshots[mask].set_index("symbol")["weight"]
        weight_lookup[d] = w

    rebal_ts = [pd.Timestamp(d) for d in rebalance_dates]

    port_records = []
    for week_date in weekly_returns.index:
        valid_rebals = [d for d in rebal_ts if d <= week_date]
        if not valid_rebals:
            continue
        rebal_date = max(valid_rebals)
        weights = weight_lookup[rebal_date]
        week_ret = weekly_returns.loc[week_date]
        common = weights.index.intersection(week_ret.dropna().index)
        if len(common) == 0:
            continue
        w = weights.reindex(common).fillna(0)
        r = week_ret.reindex(common).fillna(0)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        port_records.append({"date": week_date, "portfolio_return": float((w * r).sum())})

    if not port_records:
        raise RuntimeError("No portfolio returns computed")

    port_df = pd.DataFrame(port_records)
    if not benchmark.empty:
        merged = pd.merge(port_df, benchmark, on="date", how="left")
        merged["benchmark_return"] = pd.to_numeric(merged["benchmark_return"], errors="coerce")
    else:
        merged = port_df.copy()
        merged["benchmark_return"] = np.nan
    merged = merged.sort_values("date").reset_index(drop=True)

    full_kpis = _compute_kpis(merged, rebalance_dates, weight_lookup, rf)

    coverage_date_str = _get_coverage_start_date()
    since_coverage_kpis = None
    if coverage_date_str:
        coverage_ts = pd.Timestamp(coverage_date_str)
        if coverage_ts > merged["date"].min():
            merged_since = merged[merged["date"] >= coverage_ts].reset_index(drop=True)
            if len(merged_since) >= 2:
                since_coverage_kpis = _compute_kpis(merged_since, rebalance_dates, weight_lookup, rf)

    # Charts
    cum_port_series = (1 + merged["portfolio_return"]).cumprod() - 1
    dd_vals = (np.cumprod(1 + merged["portfolio_return"].values) /
               np.maximum.accumulate(np.cumprod(1 + merged["portfolio_return"].values))) - 1

    bench_aligned = merged.dropna(subset=["benchmark_return"]).copy()
    cum_bench_map = {}
    if not bench_aligned.empty:
        bench_curve = (1 + bench_aligned["benchmark_return"]).cumprod() - 1
        cum_bench_map = dict(zip(bench_aligned["date"], bench_curve))

    monthly_df = merged.copy()
    monthly_df["month"] = monthly_df["date"].dt.to_period("M")

    def _compound(s):
        s = s.dropna()
        return (1 + s).prod() - 1 if not s.empty else np.nan

    monthly = monthly_df.groupby("month").apply(
        lambda g: pd.Series({
            "portfolio_return": (1 + g["portfolio_return"]).prod() - 1,
            "benchmark_return": _compound(g["benchmark_return"]),
        }),
        include_groups=False,
    ).reset_index()
    monthly["month"] = monthly["month"].dt.to_timestamp()

    charts = {
        "cumulative_returns": [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "portfolio": round(float(cum_port_series.iloc[i]), 6),
                "benchmark": (
                    round(float(cum_bench_map[row["date"]]), 6)
                    if row["date"] in cum_bench_map else None
                ),
            }
            for i, (_, row) in enumerate(merged.iterrows())
        ],
        "drawdown": [
            {
                "date": merged.iloc[i]["date"].strftime("%Y-%m-%d"),
                "drawdown": round(float(dd_vals[i]), 6),
            }
            for i in range(len(dd_vals))
        ],
        "monthly_returns": [
            {
                "month": row["month"].strftime("%Y-%m"),
                "portfolio": round(float(row["portfolio_return"]), 6),
                "benchmark": (
                    round(float(row["benchmark_return"]), 6)
                    if pd.notna(row["benchmark_return"]) else None
                ),
            }
            for _, row in monthly.iterrows()
        ],
    }

    payload = {
        "source_run_id": run_id,
        "frozen_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "kpis": full_kpis,
        "kpis_since_coverage": since_coverage_kpis,
        "coverage_start_date": coverage_date_str,
        "charts": charts,
    }

    path = DATA_DIR / "performance_snapshot.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(f"  Written {path} ({path.stat().st_size / 1024:.0f} KB)")
    return payload


# ---------------------------------------------------------------------------
# FIFO lot reconstruction
# ---------------------------------------------------------------------------

def _batch_lookup_prices(conn, symbol_date_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], float]:
    if not symbol_date_pairs:
        return {}
    rows_df = pd.read_sql(text("""
        WITH pairs AS (
            SELECT unnest(:symbols) AS symbol, unnest(:dates) AS target_date
        )
        SELECT p.symbol, p.target_date,
               o.adj_close AS price
        FROM pairs p
        LEFT JOIN LATERAL (
            SELECT adj_close
            FROM ohlcv_daily
            WHERE symbol = p.symbol AND date <= CAST(p.target_date AS date)
            ORDER BY date DESC
            LIMIT 1
        ) o ON TRUE
    """), conn, params={
        "symbols": [s for s, _ in symbol_date_pairs],
        "dates": [d for _, d in symbol_date_pairs],
    })
    result = {}
    for _, row in rows_df.iterrows():
        key = (row["symbol"], str(row["target_date"]))
        if pd.notna(row.get("price")):
            result[key] = float(row["price"])
    return result


def _reconstruct_lots(run_id: str, symbols: list[str], up_to_date: str) -> dict[str, list[dict]]:
    with engine.connect() as conn:
        snapshots_df = pd.read_sql(text("""
            SELECT symbol, as_of_date, weight
            FROM portfolio_snapshots
            WHERE run_id = :run_id
              AND symbol = ANY(:symbols)
              AND as_of_date <= CAST(:up_to_date AS date)
            ORDER BY symbol, as_of_date
        """), conn, params={"run_id": run_id, "symbols": symbols, "up_to_date": up_to_date})

    if snapshots_df.empty:
        return {s: [] for s in symbols}

    price_pairs = []
    for sym in symbols:
        sym_data = snapshots_df[snapshots_df["symbol"] == sym].sort_values("as_of_date")
        for _, row in sym_data.iterrows():
            price_pairs.append((sym, str(row["as_of_date"])))

    with engine.connect() as conn:
        price_map = _batch_lookup_prices(conn, price_pairs)

    result: dict[str, list[dict]] = {}
    for sym in symbols:
        sym_data = snapshots_df[snapshots_df["symbol"] == sym].sort_values("as_of_date")
        if sym_data.empty:
            result[sym] = []
            continue

        open_lots: list[dict] = []
        closed_lots: list[dict] = []
        prev_weight = 0.0

        for _, row in sym_data.iterrows():
            current_weight = float(row["weight"])
            current_date = str(row["as_of_date"])
            price = price_map.get((sym, current_date))

            if current_weight > prev_weight + 1e-7:
                new_qty = current_weight - prev_weight
                open_lots.append({
                    "entry_date": current_date,
                    "entry_weight": round(new_qty, 6),
                    "entry_price": round(price, 2) if price else None,
                    "remaining_weight": round(new_qty, 6),
                })
            elif current_weight < prev_weight - 1e-7:
                qty_to_sell = prev_weight - current_weight
                while qty_to_sell > 1e-7 and open_lots:
                    lot = open_lots[0]
                    lot_remaining = lot["remaining_weight"]
                    if lot_remaining <= qty_to_sell + 1e-7:
                        closed_lots.append({
                            "entry_date": lot["entry_date"],
                            "entry_weight": lot["entry_weight"],
                            "entry_price": lot["entry_price"],
                            "exit_date": current_date,
                            "exit_price": round(price, 2) if price else None,
                            "remaining_weight": 0.0,
                            "sold_weight": round(lot_remaining, 6),
                            "status": "closed",
                        })
                        qty_to_sell -= lot_remaining
                        open_lots.pop(0)
                    else:
                        closed_lots.append({
                            "entry_date": lot["entry_date"],
                            "entry_weight": lot["entry_weight"],
                            "entry_price": lot["entry_price"],
                            "exit_date": current_date,
                            "exit_price": round(price, 2) if price else None,
                            "remaining_weight": round(lot_remaining - qty_to_sell, 6),
                            "sold_weight": round(qty_to_sell, 6),
                            "status": "closed",
                        })
                        lot["remaining_weight"] = round(lot_remaining - qty_to_sell, 6)
                        qty_to_sell = 0

            prev_weight = current_weight

        for lot in open_lots:
            lot["exit_date"] = None
            lot["exit_price"] = None
            lot["sold_weight"] = None
            lot["status"] = "open"

        result[sym] = open_lots + closed_lots
    return result


# ---------------------------------------------------------------------------
# 2. Holdings snapshot (with unrealized_pct)
# ---------------------------------------------------------------------------

def export_holdings(run_id: str) -> None:
    log.info("Exporting holdings snapshot...")

    with engine.connect() as conn:
        holdings_df = pd.read_sql(text("""
            SELECT symbol, as_of_date, weight, expected_return, sector, optimizer
            FROM portfolio_snapshots
            WHERE run_id = :run_id
            ORDER BY as_of_date, weight DESC
        """), conn, params={"run_id": run_id})

    if holdings_df.empty:
        raise RuntimeError(f"No holdings for run_id={run_id}")

    holdings_df["as_of_date"] = pd.to_datetime(holdings_df["as_of_date"])
    dates = sorted(holdings_df["as_of_date"].unique())
    all_symbols = sorted(holdings_df["symbol"].unique().tolist())
    log.info(f"  {len(dates)} dates, {len(all_symbols)} unique symbols")

    # Pre-fetch current prices for all (symbol, date) pairs
    price_pairs = []
    for _, row in holdings_df.iterrows():
        price_pairs.append((row["symbol"], str(row["as_of_date"].date())))
    price_pairs = list(set(price_pairs))

    with engine.connect() as conn:
        current_price_map = _batch_lookup_prices(conn, price_pairs)

    # Reconstruct lots for all symbols up to the last date
    last_date_str = str(dates[-1].date())
    log.info("  Reconstructing FIFO lots...")
    lots_map = _reconstruct_lots(run_id, all_symbols, last_date_str)

    # For each date, compute unrealized_pct per holding
    rows = []
    for date in dates:
        date_str = str(date.date())
        date_holdings = holdings_df[holdings_df["as_of_date"] == date]

        for _, h in date_holdings.iterrows():
            sym = h["symbol"]
            current_price = current_price_map.get((sym, date_str))

            # Get open lots for this symbol as of this date
            all_lots = lots_map.get(sym, [])
            open_lots_at_date = [
                lot for lot in all_lots
                if lot["status"] == "open"
                or (lot["status"] == "closed" and lot.get("exit_date") and lot["exit_date"] > date_str)
            ]
            # Filter to lots that were entered on or before this date
            open_lots_at_date = [
                lot for lot in open_lots_at_date
                if lot["entry_date"] <= date_str
            ]

            total_weight = 0.0
            weighted_unrealized = 0.0
            for lot in open_lots_at_date:
                ep = lot["entry_price"]
                rw = lot.get("remaining_weight", 0) or lot.get("entry_weight", 0)
                if ep and current_price and ep > 0:
                    lot_unrealized = (current_price - ep) / ep
                    if rw:
                        weighted_unrealized += rw * lot_unrealized
                        total_weight += rw

            unrealized_pct = round(weighted_unrealized / total_weight, 6) if total_weight > 0 else None

            rows.append({
                "symbol": sym,
                "as_of_date": date_str,
                "weight": round(float(h["weight"]), 6),
                "expected_return": round(float(h["expected_return"]), 6) if pd.notna(h["expected_return"]) else None,
                "sector": h["sector"] if pd.notna(h["sector"]) else None,
                "optimizer": h["optimizer"] if pd.notna(h["optimizer"]) else None,
                "unrealized_pct": unrealized_pct,
            })

    payload = {
        "source_run_id": run_id,
        "frozen_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "rows": rows,
    }

    json_path = DATA_DIR / "holdings_snapshot.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(f"  Written {json_path} ({json_path.stat().st_size / 1024:.0f} KB)")

    csv_path = DATA_DIR / "holdings_snapshot.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log.info(f"  Written {csv_path} ({csv_path.stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# 3. Symbol name snapshot
# ---------------------------------------------------------------------------

def export_symbol_names(run_id: str) -> None:
    log.info("Exporting symbol names...")
    with engine.connect() as conn:
        symbols = pd.read_sql(text("""
            SELECT DISTINCT symbol FROM portfolio_snapshots WHERE run_id = :run_id
        """), conn, params={"run_id": run_id})["symbol"].tolist()

        names_df = pd.read_sql(text("""
            SELECT symbol, name FROM stock_tickers WHERE symbol = ANY(:symbols)
        """), conn, params={"symbols": symbols})

    mapping = dict(zip(names_df["symbol"], names_df["name"]))

    payload = {
        "source": "stock_tickers.name",
        "symbols_requested": len(symbols),
        "symbols_resolved": len(mapping),
        "mapping": mapping,
    }

    path = DATA_DIR / "symbol_name_snapshot.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(f"  Written {path} ({path.stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# 4. Covariance matrix snapshot
# ---------------------------------------------------------------------------

def export_covariance(run_id: str) -> None:
    log.info("Exporting covariance matrix...")

    with engine.connect() as conn:
        # Get the last rebalance date
        last_date_row = conn.execute(text("""
            SELECT MAX(as_of_date) FROM portfolio_snapshots WHERE run_id = :run_id
        """), {"run_id": run_id}).fetchone()
        as_of_date = str(last_date_row[0])

        holdings = pd.read_sql(text("""
            SELECT symbol, sector
            FROM portfolio_snapshots
            WHERE run_id = :run_id AND as_of_date = :as_of_date
            ORDER BY sector, symbol
        """), conn, params={"run_id": run_id, "as_of_date": as_of_date})

        if holdings.empty:
            log.warning("  No holdings found for covariance matrix")
            return

        symbols_sorted = holdings["symbol"].tolist()
        sectors_sorted = holdings["sector"].fillna("Unknown").tolist()

        prices = pd.read_sql(text("""
            SELECT symbol, date, adj_close
            FROM ohlcv_daily
            WHERE symbol = ANY(:symbols)
              AND date BETWEEN (CAST(:as_of_date AS date) - INTERVAL '400 days') AND CAST(:as_of_date AS date)
              AND adj_close IS NOT NULL
            ORDER BY symbol, date
        """), conn, params={"symbols": symbols_sorted, "as_of_date": as_of_date})

    if prices.empty:
        log.warning("  No price data for covariance matrix")
        return

    prices["adj_close"] = prices["adj_close"].astype(float)
    price_matrix = prices.pivot(index="date", columns="symbol", values="adj_close").sort_index()

    valid_cols = price_matrix.columns[price_matrix.notna().sum() >= 60]
    price_matrix = price_matrix[valid_cols]
    price_matrix = price_matrix.tail(252)
    returns = price_matrix.pct_change(fill_method=None).iloc[1:]

    # Covariance matrix, annualized
    cov = returns.cov() * 252

    ordered = [s for s in symbols_sorted if s in cov.columns]
    ordered_sectors = [sectors_sorted[symbols_sorted.index(s)] for s in ordered]
    cov = cov.loc[ordered, ordered]

    matrix = []
    for sym in ordered:
        row_vals = []
        for sym2 in ordered:
            val = cov.loc[sym, sym2]
            row_vals.append(round(float(val), 6) if pd.notna(val) else 0)
        matrix.append(row_vals)

    payload = {
        "as_of_date": as_of_date,
        "lookback_days": 252,
        "annualized": True,
        "symbols": ordered,
        "sectors": ordered_sectors,
        "matrix": matrix,
    }

    path = DATA_DIR / "covariance_snapshot.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(f"  Written {path} ({path.stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export snapshot data from PostgreSQL")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="Run ID to export")
    args = parser.parse_args()
    run_id = args.run_id

    log.info(f"Exporting snapshots for run_id={run_id}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    export_performance(run_id)
    export_holdings(run_id)
    export_symbol_names(run_id)
    export_covariance(run_id)

    log.info("Done!")


if __name__ == "__main__":
    main()
