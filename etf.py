import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings('ignore')

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(layout="wide")

# 设置页面配置
st.set_page_config(
    page_title="ETF轮动策略分析",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化session state
if 'calculated' not in st.session_state:
    st.session_state.calculated = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'etf_name_map' not in st.session_state:
    st.session_state.etf_name_map = {}
if 'strategy_metrics' not in st.session_state:
    st.session_state.strategy_metrics = None
if 'benchmark_metrics' not in st.session_state:
    st.session_state.benchmark_metrics = None

# 定义ETF数据库
etf_database = {
    "A股市场轮动": {
        "510300": "沪深300ETF",
        "512100": "中证1000ETF",
        "510050": "上证50ETF",
        "563300": "中证2000ETF",
        "588000": "科创50ETF",
        "560050": "A50ETF",
        "563360": "A500ETF",
        "512770": "战略新兴ETF",
    },
    "行业主题轮动": {
        "512880": "证券ETF",
        "512480": "半导体ETF",
        "512010": "医药ETF",
        "159928": "消费ETF",
        "512800": "银行ETF",
        "512400": "有色金属ETF",
        "515220": "煤炭ETF",
        "512200": "房地产ETF",
        "159883": "医疗器械ETF",
        "159611": "电力ETF",
        "516970": "基建50ETF",
        "159996": "家电ETF",
        "159939": "信息技术ETF",
        "159227": "航空航天ETF",
        "517380": "创新药ETF",
        "513120": "港股创新药ETF",
        "159930": "能源ETF",
        "159546": "集成电路ETF",
        "516900": "食品饮料ETF",
        "159662": "交运ETF",
        "560190": "公用事业ETF",
        "563210": "专精特新ETF",
        "159944": "材料ETF",
        "562580": "可选消费ETF",
        "588200": "科创芯片ETF",
        "512170": "医疗ETF",
        "515880": "通信ETF",
        "159819": "人工智能ETF",
        "512690": "酒ETF",
        "515700": "新能车ETF",
        "515790": "光伏ETF",
        "562500": "机器人ETF",
        "159870": "化工ETF",
        "512710": "军工龙头",
        "516150": "稀土ETF",
        "159755": "电池ETF",
        "159869": "游戏ETF",
        "517520": "黄金股ETF",
        "513770": "港股互联网ETF",
        "560630": "机器人产业ETF",
        "560860": "工业有色ETF",
        "513750": "港股通非银ETF",
        "516860": "金融科技ETF",
        "516160": "新能源ETF",
        "159766": "旅游ETF",
        "515210": "钢铁ETF",
        "512980": "传媒ETF",
        "516510": "云计算ETF",
        "159825": "农业ETF",
        "560080": "中药ETF",
        "515400": "大数据ETF",
        "159837": "生物科技",
        "515150": "一带一路",
        "516670": "畜牧养殖",
        "159698": "粮食",
        "563180": "高股息",
        "159786": "VR",
        "516260": "物联网",
    },
    "全球市场": {
        "520830": "沙特ETF",
        "513500": "标普500ETF",
        "513030": "德国30ETF",
        "513880": "日经225ETF",
        "513310": "中韩半导体ETF",
        "513220": "中概互联ETF",
        "513080": "法国CAC40ETF",
        "520580": "新兴亚洲ETF",
        "513010": "恒生科技ETF",
        "513730": "东南亚科技ETF",
        "159735": "港股消费ETF",
        "159529": "标普消费ETF",
        "513320": "恒生新经济",
    },
    "债券红利市场": {
        "511360": "短融ETF",
        "511130": "30年国债ETF",
        "511180": "上证可转债ETF",
        "513910": "港股央企红利ETF",
    },
    "商品资产": {
        "159985": "豆粕ETF",
        "159981": "能源化工ETF",
    },
    "美林时钟": {
        "510880": "红利ETF",
        "159952": "创业板ETF",
        "513100": "纳指ETF",
        "159980": "有色ETF",
        "518880": "黄金ETF",
        "511880": "银华日利ETF",
    }
}

# 侧边栏 - 参数设置
st.sidebar.header("策略参数配置")

# 选择ETF
st.sidebar.subheader("选择轮动ETF")
selected_etfs = []

# 初始化ETF选择状态 - 默认全不勾选
for category, etfs in etf_database.items():
    for code in etfs.keys():
        if f'selected_{code}' not in st.session_state:
            st.session_state[f'selected_{code}'] = False

# 为每个类别添加ETF选择
for category, etfs in etf_database.items():
    st.sidebar.markdown(f"**{category}**")

    # 添加全选按钮
    col1, col2 = st.sidebar.columns([1, 3])
    with col1:
        if st.sidebar.button(f"全选{category}", key=f"select_all_{category}"):
            for code in etfs.keys():
                st.session_state[f'selected_{code}'] = True
            st.rerun()

    with col2:
        if st.sidebar.button(f"全不选{category}", key=f"deselect_all_{category}"):
            for code in etfs.keys():
                st.session_state[f'selected_{code}'] = False
            st.rerun()

    # 显示该类别下的ETF
    for code, name in etfs.items():
        # 使用回调函数更新状态
        def update_selected(code=code):
            st.session_state[f'selected_{code}'] = st.session_state[f'checkbox_{code}']


        # 显示ETF复选框
        checkbox = st.sidebar.checkbox(
            f"{name} ({code})",
            key=f"checkbox_{code}",
            value=st.session_state[f'selected_{code}'],
            on_change=update_selected,
            args=(code,)
        )

        if st.session_state[f'selected_{code}']:
            selected_etfs.append(code)

# 显示选择统计
selected_count = len(selected_etfs)
total_count = sum(len(etfs) for etfs in etf_database.values())
st.sidebar.info(f"已选择 {selected_count}/{total_count} 个ETF")

# 添加重置按钮
if st.sidebar.button("重置选择"):
    # 重置所有ETF选择为未选中
    for category, etfs in etf_database.items():
        for code in etfs.keys():
            st.session_state[f'selected_{code}'] = False

    # 刷新页面
    st.rerun()

# 如果用户没有选择任何ETF，使用默认组合
if not selected_etfs:
    default_etfs = ["510880", "159915", "513100", "518880"]
    st.sidebar.info("使用默认ETF组合：红利ETF(510880)、创业板ETF(159915)、纳指ETF(513100)、黄金ETF(518880)")
    # 将默认ETF添加到selected_etfs
    selected_etfs.extend(default_etfs)

# 添加警告信息
if len(selected_etfs) > 20:
    st.sidebar.warning("选择过多ETF可能导致计算缓慢，建议不超过20个")

# 时间范围选择
st.sidebar.subheader("时间范围")
end_date = datetime.today()
start_date = end_date - timedelta(days=365 * 5)  # 默认5年
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("开始日期", value=start_date)
with col2:
    end_date = st.date_input("结束日期", value=end_date)

# 策略参数
st.sidebar.subheader("策略参数")
n_days = st.sidebar.slider("得分计算窗口(天)", min_value=10, max_value=100, value=25)
trade_fee = st.sidebar.slider("交易费率(%)", min_value=0.0, max_value=0.5, value=0.01, step=0.01) / 100

# 添加买入条件开关
enable_ma_condition = st.sidebar.checkbox("启用20日均线买入条件", value=False,
                                          help="勾选后，只有当天的20日均线高于2个交易日前的20日均线时才执行买入操作")

# 添加风控阈值选项
enable_risk_control = st.sidebar.checkbox("启用风控阈值", value=False,
                                          help="勾选后，当ETF得分超过设定值时，将不被考虑买入或将被卖出")

# 如果启用风控阈值，显示阈值输入框
risk_control_threshold = 100.0
if enable_risk_control:
    risk_control_threshold = st.sidebar.number_input(
        "风控阈值(得分上限)",
        min_value=0.0,
        max_value=1000.0,
        value=100.0,
        step=10.0,
        help="当ETF得分超过此值时，将不被考虑买入或将被卖出"
    )

# 添加计算按钮
calculate_button = st.sidebar.button("开始计算策略", type="primary")


# 计算强弱得分 - 修复NaN问题
def calculate_score(srs, N=25):
    if len(srs) < N:
        return np.nan

    # 检查是否有NaN或零值
    if srs.isna().any() or (srs <= 0).any():
        return np.nan

    try:
        x = np.arange(1, N + 1)
        y = srs.values / srs.values[0]  # 归一化
        lr = LinearRegression().fit(x.reshape(-1, 1), y)
        slope = lr.coef_[0]  # 斜率
        r_squared = lr.score(x.reshape(-1, 1), y)  # R²
        return 10000 * slope * r_squared  # 综合得分
    except Exception as e:
        return np.nan


# 获取数据
@st.cache_data(ttl=3600, show_spinner="获取ETF行情数据...")
def get_etf_data(codes, start_date, end_date):
    df_list = []
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    for code in codes:
        try:
            df = ak.fund_etf_hist_em(symbol=code, period='daily',
                                     start_date=start_str, end_date=end_str, adjust='hfq')
            df.insert(0, 'code', code)
            df_list.append(df)
            time.sleep(0.5)  # 防止请求过快
        except Exception as e:
            st.warning(f"获取 {code} 数据失败: {str(e)}")

    if not df_list:
        st.error("未能获取任何ETF数据，请检查代码或网络连接")
        st.stop()

    all_df = pd.concat(df_list, ignore_index=True)
    data = all_df.pivot(index='日期', columns='code', values='收盘')[codes]
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    # 填充缺失值
    data = data.ffill().bfill()

    # 确保没有零值
    data.replace(0, np.nan, inplace=True)
    data = data.ffill().bfill()

    return data


# 计算策略
def calculate_strategy(data, n_days, trade_fee, enable_ma_condition=False,
                       enable_risk_control=False, risk_control_threshold=100.0):
    # 保存原始列名
    original_columns = data.columns.tolist()

    # 计算每日涨跌幅
    for code in original_columns:
        data[f'{code}_日收益率'] = data[code].pct_change()

    # 计算得分
    for code in original_columns:
        data[f'{code}_得分'] = data[code].rolling(n_days).apply(
            lambda x: calculate_score(x, n_days), raw=False
        )

    # 如果启用均线条件，计算20日均线
    if enable_ma_condition:
        for code in original_columns:
            data[f'{code}_ma20'] = data[code].rolling(20).mean()

    # 移除无效数据
    data = data.dropna()

    # 初始化列
    data['策略日收益率'] = 0.0
    data['持有标的'] = None
    data['交易信号'] = 0  # 0:无交易, 1:买入, -1:卖出/空仓
    data['交易原因'] = ""  # 记录交易原因
    if enable_ma_condition:
        data['均线条件满足'] = False

    # 生成交易信号
    previous_holding = None  # 初始持仓

    for i in range(n_days, len(data)):
        current_date = data.index[i]

        # 获取前一日得分并排序
        scores = {code: data[f'{code}_得分'].iloc[i - 1] for code in original_columns}
        sorted_codes = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # 选择ETF的逻辑
        selected_code = None
        trade_reason = ""

        # 寻找最佳可用ETF（满足所有条件）
        best_available_code = None
        for code in sorted_codes:
            # 如果启用风控阈值，跳过超过阈值的ETF
            if enable_risk_control and scores[code] > risk_control_threshold:
                continue

            # 如果启用均线条件，检查均线条件
            if enable_ma_condition:
                # 使用前一天的均线值（实际可获取）
                ma_latest = data[f'{code}_ma20'].iloc[i - 1]
                # 获取3个交易日前的均线值（相当于2个交易日前）
                ma_two_days_before = data[f'{code}_ma20'].iloc[i - 3]

                if ma_latest <= ma_two_days_before:
                    continue  # 均线条件不满足，跳过

            best_available_code = code
            break

        # 检查当前持仓是否需要卖出
        if previous_holding is not None:
            current_score = scores[previous_holding]

            # 如果启用风控阈值且得分超过阈值，卖出
            if enable_risk_control and current_score > risk_control_threshold:
                trade_reason = f"得分({current_score:.2f})超过风控阈值({risk_control_threshold})"
                selected_code = None
            # 如果有更好的ETF可用，且其得分高于当前持仓
            elif best_available_code is not None and best_available_code != previous_holding and \
                    scores[best_available_code] > current_score:
                trade_reason = f"发现更高得分ETF({best_available_code}, {scores[best_available_code]:.2f} > {current_score:.2f})"
                selected_code = best_available_code
            else:
                # 继续持有当前持仓
                selected_code = previous_holding
                trade_reason = f"继续持有({previous_holding})，得分({current_score:.2f})"
        else:
            # 当前空仓，使用最佳可用ETF
            selected_code = best_available_code
            if selected_code is not None:
                trade_reason = f"选择得分最高的ETF({selected_code})，得分({scores[selected_code]:.2f})"
            else:
                trade_reason = "所有ETF均不满足条件"

        # 检查是否需要交易
        trade_signal = 0
        if previous_holding != selected_code:
            if selected_code is None:
                trade_signal = -1  # 卖出
            else:
                trade_signal = 1  # 买入
            previous_holding = selected_code

        # 记录持仓和信号
        data.loc[current_date, '持有标的'] = selected_code
        data.loc[current_date, '交易信号'] = trade_signal
        data.loc[current_date, '交易原因'] = trade_reason

        # 计算当日收益率
        if selected_code is None:
            # 空仓时收益率为0（现金）
            daily_return = 0.0
        else:
            daily_return = data.loc[current_date, f'{selected_code}_日收益率']
            # 换仓时扣除费用
            if trade_signal == 1:
                daily_return -= trade_fee

        data.loc[current_date, '策略日收益率'] = daily_return

    # 计算策略净值
    data['轮动策略净值'] = (1 + data['策略日收益率']).cumprod()

    # 计算基准净值
    equal_weight_returns = data[[f'{c}_日收益率' for c in original_columns]].mean(axis=1)
    data['等权重组合净值'] = (1 + equal_weight_returns).cumprod()

    return data


# 计算性能指标 - 修复长度不一致问题
def calculate_performance_metrics(returns, benchmark_returns=None):
    # 累计收益率
    cumulative_return = (1 + returns).prod() - 1

    # 年化收益率
    annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1

    # 最大回撤
    wealth_index = (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    max_drawdown = drawdown.min()

    # 夏普比率 (假设无风险利率为0)
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)

    # 相对于基准的指标
    alpha = None
    beta = None
    if benchmark_returns is not None:
        # 确保两个序列长度相同
        min_len = min(len(returns), len(benchmark_returns))
        returns_aligned = returns.iloc[:min_len]
        benchmark_aligned = benchmark_returns.iloc[:min_len]

        # 计算alpha和beta
        cov = np.cov(returns_aligned, benchmark_aligned)
        beta = cov[0, 1] / cov[1, 1]
        alpha = returns_aligned.mean() - beta * benchmark_aligned.mean()

    return {
        '累计收益率': cumulative_return,
        '年化收益率': annualized_return,
        '最大回撤': max_drawdown,
        '夏普比率': sharpe_ratio,
        'Alpha': alpha,
        'Beta': beta
    }


# 主应用
st.title("📈 ETF轮动策略分析工具")
st.markdown("""
本工具实现基于动量得分的ETF轮动策略：
1. 计算各ETF在指定窗口期内的线性回归斜率和决定系数
2. 每日选择得分最高的ETF持有
3. 当最优ETF变化时进行换仓（扣除交易费用）
4. 可选风控阈值：当ETF得分超过设定值时，将不被考虑买入或将被卖出
""")

# 显示选中的ETF
etf_names = []
for code in selected_etfs:
    for category, etfs in etf_database.items():
        if code in etfs:
            etf_names.append(f"{etfs[code]} ({code})")
            break

st.subheader(f"轮动组合: {', '.join(etf_names)}")
st.caption(f"回测周期: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')} | "
           f"得分窗口: {n_days}天 | 交易费率: {trade_fee * 100:.2f}%")

if enable_ma_condition:
    st.info("已启用20日均线买入条件：只有当天的20日均线高于2个交易日前的20日均线时才执行买入操作")

if enable_risk_control:
    st.info(f"已启用风控阈值：当ETF得分超过{risk_control_threshold}时，将不被考虑买入或将被卖出")

# 当点击计算按钮时执行计算
if calculate_button:
    with st.spinner("正在获取ETF行情数据..."):
        st.session_state.data = get_etf_data(selected_etfs, start_date, end_date)

    with st.spinner("正在计算轮动策略..."):
        try:
            st.session_state.data = calculate_strategy(
                st.session_state.data,
                n_days,
                trade_fee,
                enable_ma_condition,
                enable_risk_control,
                risk_control_threshold
            )

            # 获取ETF名称映射
            etf_name_map = {}
            for category, etfs in etf_database.items():
                for code, name in etfs.items():
                    etf_name_map[code] = name
            st.session_state.etf_name_map = etf_name_map

            # 性能指标计算
            if '策略日收益率' in st.session_state.data.columns:
                strategy_returns = st.session_state.data['策略日收益率']
                benchmark_returns = st.session_state.data['等权重组合净值'].pct_change().dropna()

                # 确保两个序列长度相同
                min_len = min(len(strategy_returns), len(benchmark_returns))
                strategy_returns = strategy_returns.iloc[:min_len]
                benchmark_returns = benchmark_returns.iloc[:min_len]

                st.session_state.strategy_metrics = calculate_performance_metrics(strategy_returns, benchmark_returns)
                st.session_state.benchmark_metrics = calculate_performance_metrics(benchmark_returns)
            else:
                st.error("策略计算失败，请检查数据")
                st.stop()

            st.session_state.calculated = True
            st.success("策略计算完成！")
        except Exception as e:
            st.error(f"计算策略时出错: {str(e)}")
            st.stop()

# 如果已经计算过，显示结果
if st.session_state.calculated and st.session_state.data is not None:
    # 显示关键指标
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("策略最终净值", f"{st.session_state.data['轮动策略净值'].iloc[-1]:.2f}")
    col2.metric("策略年化收益", f"{st.session_state.strategy_metrics['年化收益率']:.2%}")
    col3.metric("最大回撤", f"{st.session_state.strategy_metrics['最大回撤']:.2%}")
    col4.metric("夏普比率", f"{st.session_state.strategy_metrics['夏普比率']:.2f}")

    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs(["净值曲线", "持仓分析", "回撤分析", "详细数据"])

    with tab1:
        st.subheader("净值曲线对比")

        # 准备数据
        plot_data = pd.DataFrame()
        plot_data['轮动策略'] = st.session_state.data['轮动策略净值']
        plot_data['等权重组合'] = st.session_state.data['等权重组合净值']

        for code in selected_etfs:
            name = st.session_state.etf_name_map.get(code, code)
            plot_data[name] = st.session_state.data[code] / st.session_state.data[code].iloc[0]

        # 绘制净值曲线
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in plot_data.columns:
            ax.plot(plot_data.index, plot_data[col], label=col)

        ax.set_title("净值曲线对比", fontsize=14)
        ax.set_ylabel("净值", fontsize=12)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("持仓分布")

            # 计算持仓比例
            if '持有标的' in st.session_state.data.columns:
                holdings = st.session_state.data['持有标的'].value_counts(normalize=True)
                holdings_df = pd.DataFrame({
                    'ETF': [st.session_state.etf_name_map.get(c, c) for c in holdings.index],
                    '比例': holdings.values
                })

                # 绘制饼图
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(holdings_df['比例'], labels=holdings_df['ETF'], autopct='%1.1f%%',
                       startangle=90, colors=plt.cm.Pastel1.colors)
                ax.set_title("持仓时间比例", fontsize=14)
                st.pyplot(fig)
            else:
                st.warning("持仓数据不可用")

        with col2:
            st.subheader("换仓频率")

            if '交易信号' in st.session_state.data.columns:
                # 计算换仓次数
                trade_dates = st.session_state.data[st.session_state.data['交易信号'] == 1].index
                trade_count = len(trade_dates)
                holding_period = len(st.session_state.data) / max(trade_count, 1)  # 平均持仓天数

                st.metric("总换仓次数", trade_count)
                st.metric("平均持仓天数", f"{holding_period:.1f}天")

                # 显示最近5次换仓
                if trade_count > 0:
                    st.write("最近5次换仓记录:")
                    recent_trades = []
                    for date in trade_dates[-5:]:
                        etf_code = st.session_state.data.loc[date, '持有标的']
                        etf_name = st.session_state.etf_name_map.get(etf_code, etf_code)
                        trade_reason = st.session_state.data.loc[date, '交易原因']
                        recent_trades.append({
                            '日期': date.strftime('%Y-%m-%d'),
                            '买入ETF': etf_name,
                            '原因': trade_reason
                        })
                    st.table(pd.DataFrame(recent_trades))
            else:
                st.warning("交易信号数据不可用")

    with tab3:
        st.subheader("回撤分析")

        # 计算回撤
        strategy_returns = st.session_state.data['策略日收益率']
        wealth_index = (1 + strategy_returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdown = (wealth_index - previous_peaks) / previous_peaks

        # 绘制回撤曲线
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(drawdown.index, drawdown.values, color='red', alpha=0.3)
        ax.plot(drawdown, color='darkred', linewidth=1)
        ax.set_title("策略回撤曲线", fontsize=14)
        ax.set_ylabel("回撤", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

        # 最大回撤分析
        max_dd_idx = drawdown.idxmin()
        max_dd_start = previous_peaks.idxmax()
        max_dd_end = drawdown.idxmin()
        max_dd_value = drawdown.min()

        st.write(f"**最大回撤**: {max_dd_value:.2%}")
        st.write(f"- 开始日期: {max_dd_start.strftime('%Y-%m-%d')}")
        st.write(f"- 结束日期: {max_dd_end.strftime('%Y-%m-%d')}")
        st.write(f"- 持续时间: {(max_dd_end - max_dd_start).days}天")

    with tab4:
        st.subheader("详细数据")

        # 创建详细数据表
        detail_df = pd.DataFrame(index=st.session_state.data.index)
        detail_df['轮动策略净值'] = st.session_state.data['轮动策略净值']
        detail_df['等权重组合净值'] = st.session_state.data['等权重组合净值']

        # 添加持仓信息
        if '持有标的' in st.session_state.data.columns:
            # 处理空仓状态
            detail_df['持有ETF'] = st.session_state.data['持有标的'].apply(
                lambda x: "空仓" if x is None else st.session_state.etf_name_map.get(x, x))

            # 添加交易信号
            detail_df['交易信号'] = st.session_state.data['交易信号'].apply(
                lambda x: "换仓" if x == 1 else ("卖出" if x == -1 else ""))

            # 添加交易原因
            detail_df['交易原因'] = st.session_state.data['交易原因']

            # 如果启用了均线条件，显示均线条件满足情况
            if enable_ma_condition and '均线条件满足' in st.session_state.data.columns:
                detail_df['均线条件满足'] = st.session_state.data['均线条件满足'].apply(
                    lambda x: "是" if x else "否")

        # 添加各ETF得分
        for code in selected_etfs:
            name = st.session_state.etf_name_map.get(code, code)
            detail_df[f'{name}得分'] = st.session_state.data[f'{code}_得分'].round(2)

        # 重置索引，将日期作为列
        detail_df = detail_df.reset_index()
        detail_df.rename(columns={'index': '日期'}, inplace=True)

        # 按日期降序排列（最新日期在前）
        detail_df = detail_df.sort_values('日期', ascending=False)

        # 格式化日期列
        detail_df['日期'] = detail_df['日期'].dt.strftime('%Y-%m-%d')

        # 添加数据预览选项 - 默认显示最近10天
        preview_size = st.slider("预览行数", min_value=5, max_value=50, value=10)

        # 获取最近预览行数的数据
        preview_df = detail_df.head(preview_size).copy()

        # 按最近一天的得分排序
        if len(preview_df) > 0:
            # 获取得分列
            score_columns = [col for col in preview_df.columns if '得分' in col]

            if score_columns:
                # 获取最近一天的得分
                latest_scores = preview_df.iloc[0][score_columns]

                # 按得分降序排列
                sorted_columns = latest_scores.sort_values(ascending=False).index.tolist()

                # 重新排列列顺序
                other_columns = [col for col in preview_df.columns if col not in score_columns]
                preview_df = preview_df[other_columns + sorted_columns]

                # 添加排名列
                for i, row in preview_df.iterrows():
                    scores = row[score_columns]
                    sorted_scores = scores.sort_values(ascending=False)
                    for rank, (etf, score) in enumerate(sorted_scores.items(), 1):
                        preview_df.loc[i, f'{etf}排名'] = rank

        # 显示预览数据表
        st.dataframe(preview_df)

        # 添加ETF得分排序说明
        st.caption("ETF按最近一天（第一行）的得分从高到低排序")

        # 下载按钮
        try:
            csv = detail_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="下载完整详细数据",
                data=csv,
                file_name=f"etf_轮动策略_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )
        except Exception as e:
            st.error(f"生成下载文件时出错: {str(e)}")
            st.error("请尝试减少选择的ETF数量或缩短时间范围")

        # 风控阈值分析
        if enable_risk_control:
            st.subheader("风控阈值分析")

            if '交易原因' in st.session_state.data.columns:
                # 统计风控阈值触发的次数
                risk_control_count = st.session_state.data['交易原因'].str.contains("风控阈值").sum()
                st.write(f"风控阈值触发次数: {risk_control_count}")

                # 绘制得分曲线和风控阈值线
                fig, ax = plt.subplots(figsize=(12, 6))

                # 绘制得分曲线
                for code in selected_etfs[:5]:  # 只显示前5个ETF避免过于拥挤
                    name = st.session_state.etf_name_map.get(code, code)
                    ax.plot(st.session_state.data.index, st.session_state.data[f'{code}_得分'],
                            label=f'{name}得分', alpha=0.7)

                # 添加风控阈值线
                ax.axhline(y=risk_control_threshold, color='red', linestyle='--',
                           label=f'风控阈值({risk_control_threshold})')

                ax.set_title("ETF得分与风控阈值", fontsize=14)
                ax.set_ylabel("得分", fontsize=12)
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)

# 策略说明
st.subheader("策略说明")
st.markdown("""
### ETF轮动策略原理
1. **动量得分计算**：
   - 对每个ETF计算最近N天的价格走势
   - 使用线性回归拟合价格趋势
   - 得分 = 回归斜率 × 决定系数(R²) × 10000
   - 得分越高表示趋势越强、越稳定

2. **持仓选择**：
   - 每日选择得分最高的ETF持有
   - 当最优ETF变化时进行换仓
   - 每次换仓扣除指定的交易费率

3. **20日均线买入条件（可选）**：
   - 当勾选此选项时，只有在当天的20日均线高于2个交易日前的20日均线时才执行买入操作
   - 此条件旨在过滤掉在下降趋势中的买入信号

4. **风控阈值（可选）**：
   - 当勾选此选项时，当ETF得分超过设定值时，将不被考虑买入或将被卖出
   - 此功能旨在避免在ETF过热时买入，或在ETF过热时及时卖出

5. **基准对比**：
   - 使用等权重组合作为基准（所有ETF平均配置）
   - 比较轮动策略相对于等权重组合的表现

6. **空仓逻辑**：
   - 当所有ETF得分均超过风控阈值时，策略会选择空仓
   - 当启用20日均线买入条件时，如果所有ETF都不满足均线条件，策略也会选择空仓
   - 空仓期间收益率为0，不扣除交易费用

### 使用建议
- **风控阈值设置**：
  - 建议值：80-120
  - 保守型投资者可设置较低阈值（80-100）
  - 激进型投资者可设置较高阈值（100-120）
  - 可通过回测找到最佳阈值

- **其他参数**：
  - 选择相关性较低的ETF组合效果更好
  - 适当增加得分计算窗口(25-50天)可减少换仓频率
  - 交易费率设置应考虑实际交易成本
  - 策略在趋势明显的市场环境中表现更好
  - 20日均线条件可在震荡市中减少不必要的交易
""")

# 免责声明
st.caption("""
**免责声明**：本工具提供的信息仅供参考，不构成任何投资建议。历史表现不代表未来收益，投资有风险，入市需谨慎。
""")
