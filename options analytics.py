"""
期权分析工具：
- 用 Black-Scholes 从 Yahoo Finance 提供的隐含波动率反推 Delta
- 计算"不被行权概率"（风险中性近似）
- 年化回报率
- 用历史已实现波动率(HV)近似替代 IV Rank（免费数据源拿不到逐日历史IV序列）
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, date


def dte_from_expiry(expiry_str: str) -> int:
    """expiry_str 格式如 '2026-11-20'（yfinance 格式）"""
    exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    return max((exp - date.today()).days, 0)


def _d1_d2(S, K, T, r, sigma):
    if T is None or T <= 0 or not sigma or sigma <= 0 or not S or not K or S <= 0 or K <= 0:
        return None, None
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def calc_delta(S, K, T, r, sigma, right):
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return None
    if str(right).upper().startswith("C"):
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1)


def prob_otm(S, K, T, r, sigma, right):
    """
    卖方不被行权的风险中性概率（Black-Scholes N(d2)近似）。
    这是理论定价概率，不是"真实发生概率"，仅供筛选参考。
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return None
    if str(right).upper().startswith("C"):
        return float(norm.cdf(-d2))
    return float(norm.cdf(d2))


def annualized_roi_short_option(premium_total, collateral, dte):
    """年化ROI = 权利金总额 / 占用保证金 * (365/DTE)，未考虑保证金优惠"""
    if not dte or dte <= 0 or not collateral or collateral <= 0 or premium_total is None:
        return None
    return premium_total / collateral * (365 / dte) * 100


def cash_secured_put_collateral(strike):
    return strike * 100


def covered_call_collateral(stock_price):
    return stock_price * 100


def historical_volatility(close_series: pd.Series, window=252):
    """年化历史（已实现）波动率"""
    if close_series is None or len(close_series) < 20:
        return None
    log_ret = np.log(close_series / close_series.shift(1)).dropna()
    return float(log_ret.tail(window).std() * np.sqrt(252))


def rolling_hv_series(close_series: pd.Series, window=20):
    """过去一年每日滚动20日年化历史波动率序列，用作 IV Rank 的近似对比基准"""
    if close_series is None or len(close_series) < window + 5:
        return None
    log_ret = np.log(close_series / close_series.shift(1))
    rolling_std = (log_ret.rolling(window).std() * np.sqrt(252)).dropna()
    return rolling_std


def iv_percentile_vs_history(current_iv, hv_series: pd.Series):
    """
    近似 IV Rank：当前期权隐含波动率相对于标的过去一年滚动历史波动率区间的百分位。
    【重要】这不是真正的 IV Rank（真正的IV Rank需要逐日历史隐含波动率数据，
    免费数据源拿不到），只是用已实现波动率的历史区间做近似参考，命名为 iv_rank_approx。
    """
    if current_iv is None or hv_series is None or hv_series.empty:
        return None
    lo, hi = hv_series.min(), hv_series.max()
    if hi == lo:
        return None
    pct = (current_iv - lo) / (hi - lo) * 100
    return round(max(0, min(100, pct)), 1)
