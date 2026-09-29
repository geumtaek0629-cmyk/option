import streamlit as st
import pandas as pd

import data_provider as dp
import options_analytics as oa
import strategies as strat
import stock_analysis as sa

st.set_page_config(page_title="期权筛选 & 个股分析工作台", layout="wide")
st.title("📈 期权筛选、策略扫描 & 个股四维分析")
st.caption("数据源：Yahoo Finance（免费，通常有15-20分钟延迟）。本工具不构成投资建议，仅供参考。")

with st.sidebar:
    st.header("自选股列表")
    watchlist_str = st.text_area("股票代码（英文逗号分隔）", "NVDA, AAPL, TQQQ")
    watchlist = [s.strip().upper() for s in watchlist_str.split(",") if s.strip()]
    risk_free_rate = st.number_input("无风险利率（用于Black-Scholes模型）", value=0.045, step=0.005, format="%.3f")
    st.caption("Delta / 不被行权概率均由 Black-Scholes 模型基于 Yahoo Finance 的隐含波动率反推计算")

tab1, tab2 = st.tabs(["🎯 期权筛选 & 策略扫描", "🔍 个股四维分析"])

# ==================== TAB 1：期权筛选 & 策略扫描 ====================
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        strategy_choice = st.selectbox(
            "策略类型",
            ["自定义筛选", "Wheel（现金担保卖put）", "Covered Call（备兑开仓/caller）",
             "LEAPS（长期看涨替代持股）", "穷人版备兑开仓（PMCC）"]
        )
        delta_range = st.slider("Delta 绝对值范围", 0.0, 1.0, (0.15, 0.20), 0.01)
        dte_range = st.slider("到期天数 DTE", 0, 400, (30, 45))
    with c2:
        min_iv = st.slider("最小隐含波动率 IV", 0.0, 2.0, 0.30, 0.01)
        min_volume = st.number_input("最小成交量（张）", value=500, step=100)
        min_roi = st.number_input("最小年化回报率 ROI (%)", value=30.0, step=1.0)
        min_potm = st.slider("最小不被行权概率（Black-Scholes近似）", 0.0, 1.0, 0.60, 0.01)

    run_btn = st.button("🔍 扫描期权链", type="primary", use_container_width=True)

    if run_btn:
        if not watchlist:
            st.warning("请在左侧输入至少一只股票代码")
        else:
            all_rows = []
            progress = st.progress(0, text="开始扫描...")
            for idx, symbol in enumerate(watchlist):
                progress.progress(idx / len(watchlist), text=f"正在获取 {symbol} ...")
                try:
                    t, price = dp.get_underlying_price(symbol)
                    if not price:
                        st.warning(f"{symbol}: 无法获取现价，跳过")
                        continue

                    hist = dp.get_history(symbol, period="1y")
                    hv_series = oa.rolling_hv_series(hist["Close"]) if hist is not None and not hist.empty else None

                    expirations = dp.get_expirations(t)
                    valid_expiries = [e for e in expirations
                                       if dte_range[0] <= oa.dte_from_expiry(e) <= dte_range[1]]
                    if not valid_expiries:
                        st.info(f"{symbol}: 在设定的 DTE 范围内没有可交易到期日")
                        continue

                    for exp in valid_expiries:
                        df = dp.get_option_chain(t, exp)
                        if df.empty:
                            continue
                        df = df[(df["strike"] >= 0.5 * price) & (df["strike"] <= 1.5 * price)].copy()
                        if df.empty:
                            continue

                        df["underlying_price"] = price
                        df["dte"] = df["expiry"].apply(oa.dte_from_expiry)
                        df["iv"] = df["impliedVolatility"]
                        df["delta"] = df.apply(lambda r: oa.calc_delta(
                            price, r["strike"], r["dte"] / 365, risk_free_rate, r["iv"], r["right"]), axis=1)
                        df["prob_otm"] = df.apply(lambda r: oa.prob_otm(
                            price, r["strike"], r["dte"] / 365, risk_free_rate, r["iv"], r["right"]), axis=1)
                        df["mid_price"] = df.apply(
                            lambda r: (r["bid"] + r["ask"]) / 2 if r["bid"] and r["ask"] else r.get("lastPrice"),
                            axis=1)
                        df["collateral"] = df.apply(
                            lambda r: oa.cash_secured_put_collateral(r["strike"]) if r["right"] == "P"
                            else oa.covered_call_collateral(price), axis=1)
                        df["annual_roi"] = df.apply(
                            lambda r: oa.annualized_roi_short_option(
                                r["mid_price"] * 100 if r["mid_price"] else None, r["collateral"], r["dte"]), axis=1)
                        df["iv_rank_approx"] = df["iv"].apply(lambda x: oa.iv_percentile_vs_history(x, hv_series))
                        df["volume"] = df["volume"].fillna(0)
                        df["symbol"] = symbol
                        all_rows.append(df)
                except Exception as e:
                    st.warning(f"{symbol} 处理出错：{e}")
                    continue

            progress.progress(1.0, text="完成")

            if not all_rows:
                st.error("未获取到任何期权数据。可能原因：代码错误、该标的没有期权交易、或所选 DTE 范围内无到期日。")
            else:
                full_df = pd.concat(all_rows, ignore_index=True)

                if strategy_choice.startswith("Wheel"):
                    result = strat.scan_wheel_puts(full_df, delta_range[0], delta_range[1],
                                                    dte_range[0], dte_range[1], min_roi, min_potm)
                elif strategy_choice.startswith("Covered Call"):
                    result = strat.scan_covered_calls(full_df, delta_range[0], delta_range[1],
                                                       dte_range[0], dte_range[1], min_roi, min_potm)
                elif strategy_choice.startswith("LEAPS"):
                    result = strat.scan_leaps(full_df, delta_range[0], delta_range[1], dte_range[0])
                elif strategy_choice.startswith("穷人版"):
                    result = strat.scan_poor_mans_covered_call(full_df)
                else:
                    result = full_df[
                        full_df["delta"].abs().between(delta_range[0], delta_range[1]) &
                        full_df["dte"].between(*dte_range) &
                        (full_df["iv"].fillna(0) >= min_iv) &
                        (full_df["volume"].fillna(0) >= min_volume) &
                        (full_df["annual_roi"].fillna(0) >= min_roi) &
                        (full_df["prob_otm"].fillna(0) >= min_potm)
                    ].sort_values("annual_roi", ascending=False)

                st.session_state["result_df"] = result

    if "result_df" in st.session_state:
        result = st.session_state["result_df"]
        st.subheader(f"筛选结果（共 {len(result)} 条）")
        if not result.empty:
            display_cols = [c for c in [
                "symbol", "expiry", "right", "strike", "dte", "bid", "ask", "mid_price",
                "iv", "delta", "prob_otm", "annual_roi", "volume", "openInterest", "iv_rank_approx"
            ] if c in result.columns]
            sortable = [c for c in ["annual_roi", "iv_rank_approx", "prob_otm", "dte", "volume"] if c in result.columns]
            if sortable:
                sort_col = st.selectbox("排序依据", sortable, index=0)
                result = result.sort_values(sort_col, ascending=False)
            st.dataframe(result[display_cols] if display_cols else result, use_container_width=True)
            st.download_button("下载 CSV", result.to_csv(index=False).encode("utf-8-sig"),
                                "options_scan_result.csv", "text/csv")
        else:
            st.info("没有合约满足当前筛选条件，试着放宽 Delta / ROI / 不被行权概率的范围。")
    else:
        st.info("设置好筛选条件后，点击「扫描期权链」开始。")

# ==================== TAB 2：个股四维分析 ====================
with tab2:
    st.subheader("个股四维分析：基本面 / 技术面 / 资金面 / 消息面")
    st.caption("🟢事实（来自数据） 🟡推断（基于事实的解读） 🔵假设（数据缺失时的前提说明）")

    analysis_symbol = st.selectbox("选择要分析的股票", watchlist if watchlist else ["NVDA"], key="analysis_symbol")
    analyze_btn = st.button("开始分析", type="primary")

    if analyze_btn:
        with st.spinner(f"正在分析 {analysis_symbol} ..."):
            info = dp.get_fundamentals(analysis_symbol)
            hist = dp.get_history(analysis_symbol, period="1y")

            fundamental_items = sa.analyze_fundamental(info)
            technical_items = sa.analyze_technical(hist)
            capital_items = sa.analyze_capital_flow(analysis_symbol, hist, info)
            news_items = sa.analyze_news(analysis_symbol)
            score, reasons, risks = sa.compute_score_and_risks(
                fundamental_items, technical_items, capital_items, news_items)

        badge = {"fact": "🟢 事实", "inference": "🟡 推断", "assumption": "🔵 假设"}

        def render_items(items):
            for it in items:
                st.markdown(f"**{badge.get(it['type'], it['type'])}**：{it['text']}")

        st.markdown(f"## {analysis_symbol} 综合评分：{score} / 100")
        st.caption("评分为基于公开免费数据的透明规则打分（基础分50，正/负面推断各±5），不构成投资建议")

        with st.expander("📊 基本面", expanded=True):
            render_items(fundamental_items)
        with st.expander("📈 技术面", expanded=True):
            render_items(technical_items)
        with st.expander("💰 资金面", expanded=True):
            render_items(capital_items)
        with st.expander("📰 消息面", expanded=True):
            render_items(news_items)

        st.markdown("### ⚠️ 风险提示")
        for r in risks:
            st.markdown(f"- {r}")
    else:
        st.info("选择股票后点击「开始分析」")
