"""
策略扫描逻辑（数据源无关，只依赖标准化后的 DataFrame 列名）
需要的列: right, delta, dte, annual_roi, prob_otm, strike, bid, ask, expiry
"""
import pandas as pd


def scan_wheel_puts(df, min_delta=0.15, max_delta=0.30, min_dte=30, max_dte=45,
                     min_annual_roi=20, min_potm=0.60):
    """Wheel 第一步：现金担保卖出看跌"""
    puts = df[df["right"] == "P"].copy()
    puts = puts[
        puts["delta"].abs().between(min_delta, max_delta) &
        puts["dte"].between(min_dte, max_dte) &
        (puts["annual_roi"].fillna(0) >= min_annual_roi) &
        (puts["prob_otm"].fillna(0) >= min_potm)
    ]
    return puts.sort_values("annual_roi", ascending=False)


def scan_covered_calls(df, min_delta=0.15, max_delta=0.30, min_dte=30, max_dte=45,
                        min_annual_roi=20, min_potm=0.60):
    """备兑开仓（caller）：持有正股，卖出虚值看涨"""
    calls = df[df["right"] == "C"].copy()
    calls = calls[
        calls["delta"].between(min_delta, max_delta) &
        calls["dte"].between(min_dte, max_dte) &
        (calls["annual_roi"].fillna(0) >= min_annual_roi) &
        (calls["prob_otm"].fillna(0) >= min_potm)
    ]
    return calls.sort_values("annual_roi", ascending=False)


def scan_leaps(df, min_delta=0.70, max_delta=0.90, min_dte=300):
    """LEAPS：深度实值、长期看涨，可作股票替代或 PMCC 底仓"""
    calls = df[df["right"] == "C"].copy()
    calls = calls[calls["delta"].between(min_delta, max_delta) & (calls["dte"] >= min_dte)]
    return calls.sort_values("dte", ascending=False)


def scan_poor_mans_covered_call(df, leap_min_delta=0.70, leap_min_dte=300,
                                 short_min_delta=0.15, short_max_delta=0.30,
                                 short_min_dte=25, short_max_dte=45):
    """
    穷人版备兑开仓 (PMCC)：买入长期深度实值 LEAPS 看涨替代持股，
    同时卖出近月虚值看涨收权利金。返回可能配对组合及最大理论盈利。
    """
    leaps = scan_leaps(df, leap_min_delta, 0.95, leap_min_dte)
    shorts = df[
        (df["right"] == "C") &
        df["delta"].between(short_min_delta, short_max_delta) &
        df["dte"].between(short_min_dte, short_max_dte)
    ]
    pairs = []
    for _, leap in leaps.iterrows():
        if pd.isna(leap.get("ask")):
            continue
        for _, s in shorts.iterrows():
            if pd.isna(s.get("bid")) or s["strike"] <= leap["strike"]:
                continue
            net_debit = leap["ask"] - s["bid"]
            pairs.append({
                "leap_expiry": leap["expiry"], "leap_strike": leap["strike"],
                "leap_delta": leap["delta"], "leap_ask": leap["ask"],
                "short_expiry": s["expiry"], "short_strike": s["strike"],
                "short_delta": s["delta"], "short_bid": s["bid"],
                "net_debit": round(net_debit, 2),
                "max_profit_if_called": round((s["strike"] - leap["strike"]) * 100 - net_debit * 100, 2),
            })
    return pd.DataFrame(pairs).sort_values("max_profit_if_called", ascending=False) if pairs else pd.DataFrame()


def flag_dividend_capture(df, ex_div_date):
    """标记到期日晚于除息日的合约（双重股息/股息捕捉思路的辅助标记）"""
    df = df.copy()
    if ex_div_date is None:
        df["spans_ex_div"] = False
        return df
    df["expiry_dt"] = pd.to_datetime(df["expiry"])
    df["spans_ex_div"] = df["expiry_dt"] > pd.Timestamp(ex_div_date)
    return df
