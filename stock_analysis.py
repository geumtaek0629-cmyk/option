"""
个股四维分析：基本面 / 技术面 / 资金面 / 消息面
严格区分：
  🟢 fact       事实（直接来自数据）
  🟡 inference  推断（基于事实的解读，可能因未纳入其他信息而不准确）
  🔵 assumption 假设（数据缺失或不确定时做出的前提说明）

数据来源：Yahoo Finance（免费），存在延迟、字段缺失、口径差异的可能性。
本模块产出的评分和结论均为规则化的信息整理，不构成投资建议。
"""
import numpy as np
import pandas as pd
from datetime import datetime, timezone

import data_provider as dp
import options_analytics as oa


def _fmt_pct(x, digits=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x*100:.{digits}f}%"


def _fmt_num(x, digits=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x:,.{digits}f}"


def analyze_fundamental(info: dict):
    items = []
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    items.append({"type": "fact", "text": f"所属行业：{sector} / {industry}"})

    rev_growth = info.get("revenueGrowth")
    earnings_growth = info.get("earningsGrowth")
    gross_margin = info.get("grossMargins")
    profit_margin = info.get("profitMargins")
    roe = info.get("returnOnEquity")
    debt_to_equity = info.get("debtToEquity")
    pe = info.get("trailingPE")
    forward_pe = info.get("forwardPE")

    if rev_growth is not None:
        items.append({"type": "fact", "text": f"最近一期营收同比增速：{_fmt_pct(rev_growth)}"})
    if earnings_growth is not None:
        items.append({"type": "fact", "text": f"最近一期盈利同比增速：{_fmt_pct(earnings_growth)}"})
    if gross_margin is not None:
        items.append({"type": "fact", "text": f"毛利率：{_fmt_pct(gross_margin)}"})
    if profit_margin is not None:
        items.append({"type": "fact", "text": f"净利率：{_fmt_pct(profit_margin)}"})
    if roe is not None:
        items.append({"type": "fact", "text": f"净资产收益率(ROE)：{_fmt_pct(roe)}"})
    if pe is not None:
        items.append({"type": "fact", "text": f"市盈率(TTM)：{_fmt_num(pe)}　动态市盈率：{_fmt_num(forward_pe)}"})
    if debt_to_equity is not None:
        items.append({"type": "fact", "text": f"负债权益比：{_fmt_num(debt_to_equity)}"})

    if rev_growth is not None and profit_margin is not None:
        if rev_growth > 0.15 and profit_margin > 0.10:
            items.append({"type": "inference",
                          "text": "营收保持两位数增长且净利率为正，推断当前增长具备一定盈利支撑；"
                                   "但这只是单一期数据，是否可持续需结合多个季度的趋势才能判断。"})
        elif rev_growth < 0:
            items.append({"type": "inference",
                          "text": "最近一期营收同比下滑，推断公司可能面临需求走弱或竞争加剧，"
                                   "具体原因建议查阅最新财报电话会纪要。"})
    if debt_to_equity is not None and debt_to_equity > 150:
        items.append({"type": "inference",
                      "text": "负债权益比偏高，推断财务杠杆较大，对利率变化更敏感"
                               "（假设：未与同行业平均水平比较，此处仅为绝对值观察）。"})

    summary = info.get("longBusinessSummary")
    if summary:
        items.append({"type": "fact", "text": f"业务简介摘要：{summary[:180]}..."})

    if not items:
        items.append({"type": "assumption", "text": "未能从 Yahoo Finance 获取到该股票的基本面字段，可能是数据源覆盖不足（如新股、小盘股或非美股）。"})
    return items


def analyze_technical(hist: pd.DataFrame):
    items = []
    if hist is None or hist.empty or len(hist) < 20:
        items.append({"type": "assumption", "text": "历史价格数据不足，无法计算均线趋势。"})
        return items

    close = hist["Close"]
    last_price = float(close.iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma60 = float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60 else None

    items.append({"type": "fact", "text": f"最新收盘价：{last_price:.2f}"})
    items.append({"type": "fact", "text": f"20日均线：{sma20:.2f}" +
                  (f"　60日均线：{sma60:.2f}" if sma60 else "　（数据不足60日，60日均线暂缺）")})

    if sma60:
        if last_price > sma20 > sma60:
            items.append({"type": "inference", "text": "价格位于20日线和60日线之上，且短期均线高于长期均线，推断处于中短期上升趋势。"})
        elif last_price < sma20 < sma60:
            items.append({"type": "inference", "text": "价格位于20日线和60日线之下，且短期均线低于长期均线，推断处于中短期下降趋势。"})
        else:
            items.append({"type": "inference", "text": "价格与均线出现交叉或纠缠，推断趋势不明朗，可能处于震荡或转换阶段。"})

    high_52w, low_52w = float(close.max()), float(close.min())
    pct_from_high = (last_price / high_52w - 1) * 100
    items.append({"type": "fact", "text": f"近一年区间：{low_52w:.2f} - {high_52w:.2f}，当前价格距区间高点 {pct_from_high:.1f}%"})

    hv = oa.historical_volatility(close)
    if hv is not None:
        items.append({"type": "fact", "text": f"近一年年化历史波动率（HV）：{hv*100:.1f}%"})

    return items


def analyze_capital_flow(symbol: str, hist: pd.DataFrame, info: dict):
    items = []
    if hist is not None and not hist.empty:
        latest_vol = float(hist["Volume"].iloc[-1])
        avg_vol_20d = float(hist["Volume"].tail(20).mean())
        avg_vol_60d = float(hist["Volume"].tail(60).mean()) if len(hist) >= 60 else None
        items.append({"type": "fact", "text": f"最新成交量：{latest_vol:,.0f}　20日均量：{avg_vol_20d:,.0f}" +
                      (f"　60日均量：{avg_vol_60d:,.0f}" if avg_vol_60d else "")})

        shares_out = info.get("sharesOutstanding")
        if shares_out:
            turnover = latest_vol / shares_out * 100
            items.append({"type": "fact", "text": f"当日换手率（近似，未剔除做市商/机构内部成交）：{turnover:.2f}%"})

        if avg_vol_60d:
            change = (avg_vol_20d / avg_vol_60d - 1) * 100
            if abs(change) > 20:
                direction = "放大" if change > 0 else "萎缩"
                items.append({"type": "inference",
                              "text": f"近20日均量相对60日均量{direction} {abs(change):.0f}%，"
                                       "推断近期市场关注度或交易活跃度发生明显变化，具体驱动需结合消息面判断。"})
    else:
        items.append({"type": "assumption", "text": "无法获取成交量历史数据。"})

    holders = dp.get_institutional_holders(symbol)
    if holders is not None and not holders.empty:
        top_n = holders.head(5)
        lines = []
        for _, r in top_n.iterrows():
            pct = r.get("pctHeld")
            pct_str = f"{pct*100:.2f}%" if pct is not None else "N/A"
            lines.append(f"{r.get('Holder', 'N/A')} {pct_str}")
        items.append({"type": "fact", "text": "机构持仓前5大（Yahoo Finance 最新披露快照）：" + "；".join(lines)})
        items.append({"type": "assumption",
                      "text": "免费数据源只提供机构持仓的最新快照，无法获取逐季度变化趋势，"
                               "因此无法判断机构近期是增持还是减持；如需趋势请查阅历史13F披露文件对比。"})
    else:
        items.append({"type": "assumption", "text": "未获取到机构持仓数据（可能是数据源限制或该股票非美股主板）。"})

    return items


def analyze_news(symbol: str):
    items = []
    news = dp.get_news(symbol)
    if not news:
        items.append({"type": "assumption", "text": "未获取到近期相关新闻，可能是数据源覆盖限制。"})
    for n in news:
        pub = n.get("published")
        if isinstance(pub, (int, float)):
            pub_str = datetime.fromtimestamp(pub, tz=timezone.utc).strftime("%Y-%m-%d")
        elif isinstance(pub, str):
            pub_str = pub[:10]
        else:
            pub_str = "N/A"
        if n.get("title"):
            items.append({"type": "fact", "text": f"[{pub_str}] {n.get('title')}（来源：{n.get('publisher', 'N/A')}）"})

    earnings_date = dp.get_next_earnings_date(symbol)
    if earnings_date is not None and str(earnings_date) != "[]":
        items.append({"type": "fact", "text": f"下一次财报预期日期：{earnings_date}"})
        items.append({"type": "inference",
                      "text": "临近财报窗口期股价波动率通常会上升，推断该时间点前后可能出现较大波动，但方向无法预判。"})

    return items


def compute_score_and_risks(fundamental_items, technical_items, capital_items, news_items):
    """
    简单加权打分（基础分50），透明规则：
    每个维度里出现的正面/负面推断分别 +5/-5。
    这是一个粗粒度的规则打分，仅用于四个维度的概览对比，不是量化模型，不构成投资建议。
    """
    score = 50
    reasons = []

    def scan(items, positive_kw, negative_kw, weight=5):
        nonlocal score
        for it in items:
            if it["type"] != "inference":
                continue
            text = it["text"]
            if any(k in text for k in positive_kw):
                score += weight
                reasons.append(f"+{weight}：{text[:50]}...")
            if any(k in text for k in negative_kw):
                score -= weight
                reasons.append(f"-{weight}：{text[:50]}...")

    scan(fundamental_items, ["增长具备一定盈利支撑"], ["需求走弱", "财务杠杆较大"])
    scan(technical_items, ["上升趋势"], ["下降趋势"])
    scan(capital_items, ["放大"], ["萎缩"])

    score = max(0, min(100, score))

    risks = [
        "本评分基于公开免费数据的规则打分，维度有限、逻辑简单，不构成投资建议。",
        "Yahoo Finance 数据可能存在延迟、字段缺失或口径差异，使用前建议与官方财报/公告核对。",
    ]
    if any("财报" in it["text"] for it in news_items):
        risks.append("近期临近财报窗口期，波动率可能显著上升，注意仓位管理。")

    return score, reasons, risks
