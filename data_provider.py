"""
数据源：Yahoo Finance（通过 yfinance 库，免费，无需 API Key）

关于 Barchart：
Barchart 没有面向个人的免费公开 API，付费版 OnDemand API 需要单独申请 API Key，
接口协议和字段名都和 Yahoo Finance 不同。如果你之后购买了 Barchart 的付费订阅，
只需要重写这一个文件里的函数（保持返回的字段名/DataFrame结构一致），
app.py / strategies.py / stock_analysis.py 都不需要改动。

注意：yfinance 是对 Yahoo 网页/内部接口的非官方封装，字段结构可能随 Yahoo 改版而变化，
如果某个字段取不到，大概率是 yfinance 版本和 Yahoo 接口不匹配，建议先尝试
`pip install -U yfinance` 更新到最新版本。
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone


def get_ticker(symbol: str):
    return yf.Ticker(symbol)


def get_underlying_price(symbol: str):
    t = get_ticker(symbol)
    price = None
    try:
        price = t.fast_info.get("lastPrice") or t.fast_info.get("last_price")
    except Exception:
        pass
    if not price:
        hist = t.history(period="5d")
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
    return t, price


def get_expirations(t):
    """返回该标的所有可交易到期日，格式 'YYYY-MM-DD'"""
    try:
        return list(t.options)
    except Exception:
        return []


def get_option_chain(t, expiry: str) -> pd.DataFrame:
    """返回该到期日的 calls+puts 合并 DataFrame，含 right 列区分 C/P"""
    chain = t.option_chain(expiry)
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    calls["right"] = "C"
    puts["right"] = "P"
    df = pd.concat([calls, puts], ignore_index=True)
    df["expiry"] = expiry
    return df


def get_history(symbol: str, period="1y") -> pd.DataFrame:
    t = get_ticker(symbol)
    return t.history(period=period)


def get_fundamentals(symbol: str) -> dict:
    """注意：t.info 有时会比较慢（内部多次请求），并且字段可能因股票类型不同而不同"""
    t = get_ticker(symbol)
    try:
        return t.info or {}
    except Exception:
        return {}


def get_institutional_holders(symbol: str):
    t = get_ticker(symbol)
    try:
        return t.institutional_holders
    except Exception:
        return None


def get_major_holders(symbol: str):
    t = get_ticker(symbol)
    try:
        return t.major_holders
    except Exception:
        return None


def get_news(symbol: str, limit=8):
    """
    yfinance 不同版本的 news 结构不完全一致，这里做了兼容处理。
    只提取标题/来源/时间/链接，不抓取正文（避免版权问题，也没必要）。
    """
    t = get_ticker(symbol)
    try:
        news = t.news or []
    except Exception:
        news = []
    rows = []
    for n in news[:limit]:
        content = n.get("content", n) if isinstance(n, dict) else {}
        if isinstance(content, dict) and "title" in content:
            title = content.get("title")
            publisher = (content.get("provider") or {}).get("displayName")
            pub_date = content.get("pubDate")
            link = (content.get("canonicalUrl") or {}).get("url")
        else:
            title = n.get("title")
            publisher = n.get("publisher")
            pub_date = n.get("providerPublishTime")
            link = n.get("link")
        rows.append({"title": title, "publisher": publisher, "published": pub_date, "link": link})
    return rows


def get_dividends(symbol: str):
    t = get_ticker(symbol)
    try:
        return t.dividends
    except Exception:
        return pd.Series(dtype=float)


def get_next_earnings_date(symbol: str):
    t = get_ticker(symbol)
    try:
        cal = t.calendar
        if isinstance(cal, dict):
            return cal.get("Earnings Date")
        return cal
    except Exception:
        return None
