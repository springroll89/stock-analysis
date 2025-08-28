#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一体化股票放量突破分析系统 - 实时版
整合数据获取和分析流程，自动获取最新交易数据
版本: 2.1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import baostock as bs
import time
import warnings
import os
import json

warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="实时放量突破分析系统",
    page_icon="🚀",
    layout="wide"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .scan-progress {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .breakout-alert {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


class IntegratedStockAnalyzer:
    """一体化股票分析器"""

    def __init__(self):
        self.login_status = False
        self.cache_dir = "stock_cache"
        self.ensure_cache_dir()

    def ensure_cache_dir(self):
        """确保缓存目录存在"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def login(self):
        """登录Baostock系统"""
        if not self.login_status:
            lg = bs.login()
            if lg.error_code == '0':
                self.login_status = True
                return True
            else:
                st.error(f"Baostock登录失败: {lg.error_msg}")
                return False
        return True

    def logout(self):
        """登出Baostock系统"""
        if self.login_status:
            bs.logout()
            self.login_status = False

    def format_stock_code(self, code):
        """格式化股票代码"""
        code_str = str(code).zfill(6)
        if code_str.startswith('6'):
            return f"sh.{code_str}", code_str
        else:
            return f"sz.{code_str}", code_str

    def get_trading_dates(self, days_back=30):
        """获取最近的交易日期"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        return start_date, end_date

    @st.cache_data(ttl=300)  # 5分钟缓存
    def get_stock_list(_self, market_filter='all'):
        """获取股票列表"""
        if not _self.login():
            return []

        try:
            # 获取所有股票基本信息
            rs = bs.query_stock_basic()

            stock_list = []
            while (rs.error_code == '0') & rs.next():
                row = rs.get_row_data()
                # 只获取上市的股票
                if len(row) >= 6 and row[5] == '1':  # status = 1 表示上市
                    code = row[0]
                    name = row[1]
                    stock_type = row[4] if len(row) > 4 else '1'

                    # 根据市场筛选
                    if market_filter == 'main':
                        # 主板
                        if code.startswith('sh.60') or code.startswith('sz.00'):
                            if not code.startswith('sz.002'):  # 排除中小板
                                stock_list.append({'code': code, 'name': name})
                    elif market_filter == 'cyb':
                        # 创业板
                        if code.startswith('sz.30'):
                            stock_list.append({'code': code, 'name': name})
                    elif market_filter == 'kcb':
                        # 科创板
                        if code.startswith('sh.68'):
                            stock_list.append({'code': code, 'name': name})
                    elif market_filter == 'all':
                        # 所有A股
                        if stock_type == '1':  # type=1表示股票
                            stock_list.append({'code': code, 'name': name})

            return stock_list

        except Exception as e:
            st.error(f"获取股票列表失败: {e}")
            return []

    def calculate_volume_ratio(self, df):
        """计算放量倍数"""
        if len(df) < 2:
            return None

        # 确保按日期排序
        df = df.sort_values('date')

        # 计算前5日平均成交量
        df['avg_volume_5'] = df['volume'].rolling(window=5, min_periods=1).mean().shift(1)

        # 计算放量倍数
        df['volume_ratio'] = df['volume'] / df['avg_volume_5']

        return df

    @st.cache_data(ttl=300)  # 5分钟缓存
    def scan_single_stock(_self, code, start_date, end_date, min_volume_ratio=3.0, min_change_pct=0,
                          max_change_pct=10.0, check_confirmation=False):
        """扫描单个股票的放量突破情况"""
        try:
            # 获取K线数据
            rs = bs.query_history_k_data_plus(
                code,
                "date,open,high,low,close,preclose,volume,amount,pctChg,peTTM,pbMRQ",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2"  # 前复权
            )

            if rs.error_code != '0':
                return None

            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                return None

            # 转换为DataFrame
            df = pd.DataFrame(data_list, columns=rs.fields)

            # 数据类型转换
            numeric_cols = ['open', 'high', 'low', 'close', 'preclose',
                            'volume', 'amount', 'pctChg', 'peTTM', 'pbMRQ']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 计算放量倍数
            df = _self.calculate_volume_ratio(df)

            if df is None or df.empty:
                return None

            # 如果需要确认突破，扫描所有符合条件的放量日
            if check_confirmation:
                # 找出所有符合放量条件的日期
                breakout_mask = (
                        (df['volume_ratio'] >= min_volume_ratio) &
                        (df['pctChg'] >= min_change_pct) &
                        (df['pctChg'] <= max_change_pct) &
                        (pd.notna(df['volume_ratio']))
                )

                breakout_indices = df[breakout_mask].index.tolist()

                # 从最近的放量日开始检查
                for idx in reversed(breakout_indices):
                    # 确保有足够的后续数据（至少2天）
                    if idx + 2 >= len(df):
                        continue

                    breakout_row = df.iloc[idx]
                    breakout_close = breakout_row['close']

                    # 检查后续2-5天（或到数据结束）
                    days_to_check = min(5, len(df) - idx - 1)
                    confirmed = False
                    confirmation_day = None

                    for i in range(2, days_to_check + 1):
                        if idx + i < len(df):
                            future_row = df.iloc[idx + i]
                            # 如果任意一天收盘价站上放量日收盘价
                            if future_row['close'] >= breakout_close:
                                confirmed = True
                                confirmation_day = i
                                break

                    # 如果找到确认的突破
                    if confirmed:
                        # 计算站稳后的表现
                        latest = df.iloc[-1]
                        days_since_breakout = len(df) - idx - 1
                        price_change_pct = ((latest['close'] - breakout_close) / breakout_close) * 100

                        return {
                            'date': breakout_row['date'],  # 放量日期
                            'volume_ratio': round(breakout_row['volume_ratio'], 2),
                            'pctChg': round(breakout_row['pctChg'], 2),
                            'close': round(breakout_row['close'], 2),
                            'amount': round(breakout_row['amount'] / 1e8, 2),
                            'pe': round(breakout_row['peTTM'], 2) if pd.notna(breakout_row['peTTM']) else None,
                            'pb': round(breakout_row['pbMRQ'], 2) if pd.notna(breakout_row['pbMRQ']) else None,
                            'confirmed': True,  # 标记为已确认
                            'confirmation_day': confirmation_day,  # 第几天确认
                            'days_since': days_since_breakout,  # 距今天数
                            'price_change': round(price_change_pct, 2)  # 突破后涨幅
                        }

                # 如果没有找到确认的突破，返回None
                return None

            else:
                # 原逻辑：只检查最新一天
                latest = df.iloc[-1]

                # 判断是否满足放量突破条件
                if (pd.notna(latest['volume_ratio']) and
                        latest['volume_ratio'] >= min_volume_ratio and
                        latest['pctChg'] >= min_change_pct and
                        latest['pctChg'] <= max_change_pct):
                    return {
                        'date': latest['date'],
                        'volume_ratio': round(latest['volume_ratio'], 2),
                        'pctChg': round(latest['pctChg'], 2),
                        'close': round(latest['close'], 2),
                        'amount': round(latest['amount'] / 1e8, 2),
                        'pe': round(latest['peTTM'], 2) if pd.notna(latest['peTTM']) else None,
                        'pb': round(latest['pbMRQ'], 2) if pd.notna(latest['pbMRQ']) else None,
                        'confirmed': False  # 标记为未确认
                    }

            return None

        except Exception as e:
            return None

    def scan_market_realtime(self, market_filter='all', days_back=30,
                             min_volume_ratio=3.0, min_change_pct=0, max_change_pct=10.0,
                             check_confirmation=False, progress_callback=None):
        """实时扫描市场放量突破股票"""
        if not self.login():
            return pd.DataFrame()

        # 获取股票列表
        stock_list = self.get_stock_list(market_filter)

        if not stock_list:
            st.error("未获取到股票列表")
            return pd.DataFrame()

        # 获取交易日期
        start_date, end_date = self.get_trading_dates(days_back)

        # 扫描结果
        breakout_stocks = []
        total = len(stock_list)

        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, stock_info in enumerate(stock_list):
            # 更新进度
            progress = (idx + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"正在扫描 {stock_info['code']} {stock_info['name']} ({idx + 1}/{total})")

            # 扫描股票
            result = self.scan_single_stock(
                stock_info['code'],
                start_date,
                end_date,
                min_volume_ratio,
                min_change_pct,
                max_change_pct,
                check_confirmation  # 传递确认参数
            )

            if result:
                result['code'] = stock_info['code'].split('.')[1]
                result['name'] = stock_info['name']
                # 获取行业信息
                result['industry'] = self.get_industry_info(result['code'])
                breakout_stocks.append(result)

                # 实时显示发现的股票
                if progress_callback:
                    progress_callback(result)

            # 避免请求过快
            if idx % 50 == 0 and idx > 0:
                time.sleep(1)

        # 清理进度条
        progress_bar.empty()
        status_text.empty()

        # 转换为DataFrame
        if breakout_stocks:
            df_result = pd.DataFrame(breakout_stocks)

            # 根据是否有确认字段调整列顺序
            if check_confirmation and 'confirmed' in df_result.columns:
                # 包含确认信息的列顺序
                columns_order = ['code', 'name', 'industry', 'date', 'volume_ratio', 'pctChg',
                                 'close', 'amount', 'confirmed', 'confirmation_day', 'days_since',
                                 'price_change', 'pe', 'pb']
                # 过滤存在的列
                columns_order = [col for col in columns_order if col in df_result.columns]
                df_result = df_result[columns_order]

                # 重命名列
                rename_dict = {
                    'code': '代码',
                    'name': '名称',
                    'industry': '行业',
                    'date': '放量日期',
                    'volume_ratio': '放量倍数',
                    'pctChg': '当日涨幅%',
                    'close': '放量收盘价',
                    'amount': '成交额(亿)',
                    'confirmed': '已确认',
                    'confirmation_day': '确认天数',
                    'days_since': '距今天数',
                    'price_change': '突破后涨幅%',
                    'pe': '市盈率',
                    'pb': '市净率'
                }
                df_result.rename(columns=rename_dict, inplace=True)
            else:
                # 原有列顺序
                columns_order = ['code', 'name', 'industry', 'date', 'volume_ratio', 'pctChg',
                                 'close', 'amount', 'pe', 'pb']
                df_result = df_result[columns_order]
                # 重命名列
                df_result.columns = ['代码', '名称', '行业', '日期', '放量倍数', '涨幅%',
                                     '收盘价', '成交额(亿)', '市盈率', '市净率']

            return df_result

        return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def get_industry_info(_self, code):
        """获取股票行业信息"""
        if not _self.login():
            return "未知"

        try:
            bs_code, _ = _self.format_stock_code(code)
            rs = bs.query_stock_industry()

            while (rs.error_code == '0') & rs.next():
                row = rs.get_row_data()
                if row[1] == bs_code:  # code字段
                    return row[3] if len(row) > 3 else "未知"  # industry字段

            # 根据代码判断板块
            if str(code).startswith('300'):
                return "创业板"
            elif str(code).startswith('688'):
                return "科创板"

            return "其他"
        except:
            return "未知"

    def analyze_breakout_stocks(self, df):
        """分析放量突破股票"""
        if df.empty:
            return {}

        analysis = {
            'total_count': len(df),
            'avg_volume_ratio': df['放量倍数'].mean(),
            'avg_change_pct': df['涨幅%'].mean(),
            'total_amount': df['成交额(亿)'].sum(),
            'top_volume': df.nlargest(10, '放量倍数'),
            'top_change': df.nlargest(10, '涨幅%'),
        }

        # 如果行业列不存在，添加行业信息（向后兼容）
        if '行业' not in df.columns:
            df['行业'] = df['代码'].apply(self.get_industry_info)

        # 行业分布
        industry_dist = df.groupby('行业').agg({
            '代码': 'count',
            '放量倍数': 'mean',
            '涨幅%': 'mean',
            '成交额(亿)': 'sum'
        }).round(2)

        industry_dist.columns = ['股票数量', '平均放量', '平均涨幅', '总成交额']
        analysis['industry_distribution'] = industry_dist.sort_values('股票数量', ascending=False)

        return analysis

    def save_scan_results(self, df, scan_params):
        """保存扫描结果"""
        if df.empty:
            return None

        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"放量突破_{scan_params['market']}_{timestamp}.csv"
        filepath = os.path.join(self.cache_dir, filename)

        # 保存CSV（使用GBK编码以兼容Excel）
        try:
            df.to_csv(filepath, index=False, encoding='gbk')
        except:
            # 如果GBK失败，使用GB18030（支持更多字符）
            try:
                df.to_csv(filepath, index=False, encoding='gb18030')
            except:
                # 最后fallback到utf-8-sig
                df.to_csv(filepath, index=False, encoding='utf-8-sig')

        # 保存扫描参数
        params_file = filepath.replace('.csv', '_params.json')
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(scan_params, f, ensure_ascii=False, indent=2)

        return filepath

    def load_history_results(self):
        """加载历史扫描结果"""
        history = []

        if not os.path.exists(self.cache_dir):
            return history

        for file in os.listdir(self.cache_dir):
            if file.endswith('.csv') and file.startswith('放量突破_'):
                filepath = os.path.join(self.cache_dir, file)
                params_file = filepath.replace('.csv', '_params.json')

                # 获取文件信息
                file_stat = os.stat(filepath)
                file_time = datetime.fromtimestamp(file_stat.st_mtime)

                # 读取参数
                params = {}
                if os.path.exists(params_file):
                    with open(params_file, 'r', encoding='utf-8') as f:
                        params = json.load(f)

                history.append({
                    'filename': file,
                    'filepath': filepath,
                    'time': file_time,
                    'params': params,
                    'size': file_stat.st_size
                })

        # 按时间排序
        history.sort(key=lambda x: x['time'], reverse=True)
        return history


def main():
    """主函数"""
    st.markdown('<h1 class="main-header">🚀 实时放量突破分析系统</h1>', unsafe_allow_html=True)

    # 初始化分析器
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = IntegratedStockAnalyzer()

    analyzer = st.session_state.analyzer

    # 侧边栏设置
    with st.sidebar:
        st.markdown("### ⚙️ 扫描设置")

        # 市场选择
        market_options = {
            'all': '全部A股',
            'main': '沪深主板',
            'cyb': '创业板',
            'kcb': '科创板'
        }

        market_filter = st.selectbox(
            "选择市场",
            options=list(market_options.keys()),
            format_func=lambda x: market_options[x],
            index=0
        )

        st.markdown("---")

        # 扫描参数
        st.markdown("### 📊 扫描参数")

        days_back = st.slider(
            "扫描天数",
            min_value=5,
            max_value=60,
            value=30,
            help="扫描最近N天的数据"
        )

        min_volume_ratio = st.slider(
            "最小放量倍数",
            min_value=2.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="成交量相对于5日均量的倍数"
        )

        col1, col2 = st.columns(2)
        with col1:
            min_change_pct = st.number_input(
                "最小涨幅(%)",
                min_value=-10.0,
                max_value=20.0,
                value=0.0,
                step=0.5,
                help="涨幅下限"
            )

        with col2:
            max_change_pct = st.number_input(
                "最大涨幅(%)",
                min_value=-10.0,
                max_value=20.0,
                value=10.0,
                step=0.5,
                help="涨幅上限，可排除涨停板"
            )

        max_volume_ratio = st.slider(
            "最大放量倍数(排除异常)",
            min_value=5.0,
            max_value=20.0,
            value=10.0,
            step=1.0,
            help="排除放量过大的异常股票"
        )

        st.markdown("---")

        # 突破确认设置
        st.markdown("### 🎯 突破确认")

        check_confirmation = st.checkbox(
            "启用突破确认",
            value=False,
            help="筛选放量后2-5天内股价能站稳的股票"
        )

        if check_confirmation:
            st.info("""
            **突破确认说明：**
            - 找出历史放量突破的股票
            - 检查放量后2-5天的走势
            - 筛选股价站稳在放量日收盘价之上的股票
            - 显示突破后的表现
            """)

        # 筛选参数说明
        with st.expander("💡 参数使用说明"):
            st.markdown("""
            **涨幅筛选的作用：**
            - **最小涨幅**：筛选上涨的股票，排除下跌股
            - **最大涨幅**：排除涨停板，找到还有上涨空间的股票

            **建议设置：**
            - 寻找启动信号：最小2%，最大7%
            - 排除涨停板：最大设为9%（主板）或19%（创业板）
            - 全范围扫描：最小-10%，最大20%

            **放量倍数说明：**
            - 3倍以上：温和放量，较为可靠
            - 5倍以上：明显放量，关注度高
            - 10倍以上：异常放量，需谨慎
            """)

        # 显示当前筛选条件汇总
        st.markdown("### 📋 当前筛选条件")
        condition_text = f"""
        **市场：** {market_options[market_filter]}  
        **放量：** {min_volume_ratio}倍 - {max_volume_ratio}倍  
        **涨幅：** {min_change_pct}% - {max_change_pct}%  
        **扫描：** 最近{days_back}天
        """

        if check_confirmation:
            condition_text += "\n**模式：** 🎯 突破确认（放量后股价站稳）"

        st.info(condition_text)

        st.markdown("---")

        # 自动刷新设置
        st.markdown("### 🔄 自动刷新")

        auto_refresh = st.checkbox("开启自动刷新", value=False)

        if auto_refresh:
            refresh_interval = st.selectbox(
                "刷新间隔",
                options=[5, 10, 15, 30, 60],
                format_func=lambda x: f"{x}分钟",
                index=2
            )

            st.info(f"将每{refresh_interval}分钟自动刷新数据")

        st.markdown("---")

        # 历史记录
        st.markdown("### 📂 历史记录")

        history = analyzer.load_history_results()

        if history:
            st.write(f"共有 {len(history)} 条历史记录")

            # 显示最近的记录
            for record in history[:5]:
                with st.expander(f"📄 {record['filename']}"):
                    st.write(f"时间: {record['time'].strftime('%Y-%m-%d %H:%M')}")
                    if record['params']:
                        st.write(f"市场: {record['params'].get('market', '未知')}")
                        st.write(f"参数: 放量{record['params'].get('min_volume_ratio', 0)}倍以上")

                    if st.button(f"加载", key=f"load_{record['filename']}"):
                        # 尝试多种编码读取
                        try:
                            # 首先尝试GBK（Windows Excel默认）
                            df_history = pd.read_csv(record['filepath'], encoding='gbk')
                        except:
                            try:
                                # 然后尝试GB18030
                                df_history = pd.read_csv(record['filepath'], encoding='gb18030')
                            except:
                                try:
                                    # 尝试UTF-8
                                    df_history = pd.read_csv(record['filepath'], encoding='utf-8-sig')
                                except:
                                    # 最后尝试默认编码
                                    df_history = pd.read_csv(record['filepath'])

                        st.session_state['scan_results'] = df_history
                        st.success(f"已加载历史数据")

    # 主界面
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["📡 实时扫描", "📊 分析结果", "📈 市场热力图"])

    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("扫描市场", market_options[market_filter])
        with col2:
            if check_confirmation:
                st.metric("扫描模式", "突破确认模式", help="筛选放量后股价站稳的股票")
            else:
                st.metric("扫描参数", f"放量{min_volume_ratio}倍 | 涨幅{min_change_pct}%-{max_change_pct}%")
        with col3:
            st.metric("当前时间", datetime.now().strftime('%H:%M:%S'))

        # 扫描按钮
        if st.button("🔍 开始扫描", type="primary", use_container_width=True):

            # 显示扫描动画
            with st.spinner(f"正在扫描{market_options[market_filter]}..."):

                # 创建实时显示区域
                realtime_container = st.container()
                found_stocks = []

                def progress_callback(stock):
                    """实时显示发现的股票"""
                    found_stocks.append(stock)
                    with realtime_container:
                        st.success(f"🎯 发现: {stock['code']} {stock['name']} - "
                                   f"放量{stock['volume_ratio']}倍，涨幅{stock['pctChg']}%")

                # 执行扫描
                df_results = analyzer.scan_market_realtime(
                    market_filter=market_filter,
                    days_back=days_back,
                    min_volume_ratio=min_volume_ratio,
                    min_change_pct=min_change_pct,
                    max_change_pct=max_change_pct,
                    check_confirmation=check_confirmation,  # 传递突破确认参数
                    progress_callback=None  # 暂时不使用回调以简化显示
                )

                # 过滤异常放量
                if not df_results.empty:
                    df_results = df_results[df_results['放量倍数'] <= max_volume_ratio]

                # 保存结果
                st.session_state['scan_results'] = df_results
                st.session_state['scan_time'] = datetime.now()
                st.session_state['scan_params'] = {
                    'market': market_filter,
                    'days_back': days_back,
                    'min_volume_ratio': min_volume_ratio,
                    'min_change_pct': min_change_pct,
                    'max_change_pct': max_change_pct,
                    'max_volume_ratio': max_volume_ratio,
                    'check_confirmation': check_confirmation  # 保存确认参数
                }

                # 保存到文件
                if not df_results.empty:
                    filepath = analyzer.save_scan_results(df_results, st.session_state['scan_params'])
                    if filepath:
                        if check_confirmation:
                            st.success(f"✅ 扫描完成！发现 {len(df_results)} 只已确认突破的股票")
                            st.info("提示：这些股票在放量后2-5天内股价成功站稳，突破更为可靠")
                        else:
                            st.success(f"✅ 扫描完成！发现 {len(df_results)} 只放量突破股票")

            # 登出
            analyzer.logout()

        # 显示扫描结果
        if 'scan_results' in st.session_state:
            df_results = st.session_state['scan_results']

            if not df_results.empty:
                st.markdown("---")
                st.markdown(f"### 🎯 放量突破股票 ({len(df_results)}只)")

                # 显示统计信息
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    avg_volume = df_results['放量倍数'].mean()
                    st.metric("平均放量", f"{avg_volume:.2f}倍")

                with col2:
                    avg_change = df_results['涨幅%'].mean()
                    st.metric("平均涨幅", f"{avg_change:.2f}%")

                with col3:
                    total_amount = df_results['成交额(亿)'].sum()
                    st.metric("总成交额", f"{total_amount:.1f}亿")

                with col4:
                    max_volume = df_results['放量倍数'].max()
                    st.metric("最大放量", f"{max_volume:.1f}倍")

                # 排序选项
                sort_options = {
                    '放量倍数': '放量倍数',
                    '涨幅%': '涨幅%',
                    '成交额(亿)': '成交额(亿)',
                    '市盈率': '市盈率'
                }

                col1, col2 = st.columns([3, 1])
                with col1:
                    sort_by = st.selectbox(
                        "排序方式",
                        options=list(sort_options.keys()),
                        index=0
                    )
                with col2:
                    sort_ascending = st.checkbox("升序", value=False)

                # 显示数据表
                df_display = df_results.sort_values(sort_by, ascending=sort_ascending)

                # 检查是否有确认列
                has_confirmation = '已确认' in df_display.columns

                # 格式化显示
                format_dict = {
                    '放量倍数': '{:.1f}',
                    '涨幅%': '{:.2f}',
                    '当日涨幅%': '{:.2f}',
                    '收盘价': '{:.2f}',
                    '放量收盘价': '{:.2f}',
                    '成交额(亿)': '{:.2f}',
                    '市盈率': lambda x: f'{x:.2f}' if pd.notna(x) else '-',
                    '市净率': lambda x: f'{x:.2f}' if pd.notna(x) else '-'
                }

                # 如果有确认相关列，添加格式化
                if has_confirmation:
                    format_dict.update({
                        '已确认': lambda x: '✅' if x else '❌',
                        '确认天数': lambda x: f'第{int(x)}天' if pd.notna(x) else '-',
                        '距今天数': lambda x: f'{int(x)}天' if pd.notna(x) else '-',
                        '突破后涨幅%': '{:.2f}'
                    })

                df_styled = df_display.style.format(format_dict)

                # 高亮显示
                df_styled = df_styled.highlight_max(subset=['放量倍数'], color='lightgreen')

                # 根据列名调整高亮
                if '涨幅%' in df_display.columns:
                    df_styled = df_styled.highlight_max(subset=['涨幅%'], color='lightyellow')
                elif '当日涨幅%' in df_display.columns:
                    df_styled = df_styled.highlight_max(subset=['当日涨幅%'], color='lightyellow')

                # 如果有突破后涨幅列，也高亮显示
                if '突破后涨幅%' in df_display.columns:
                    # 正值用绿色，负值用红色
                    def color_negative_red(val):
                        try:
                            num_val = float(val)
                            color = 'lightcoral' if num_val < 0 else 'lightgreen' if num_val > 0 else ''
                            return f'background-color: {color}'
                        except:
                            return ''

                    df_styled = df_styled.applymap(color_negative_red, subset=['突破后涨幅%'])

                if '市盈率' in df_display.columns:
                    # 过滤掉'-'值，只对数值进行高亮
                    numeric_pe = pd.to_numeric(df_display['市盈率'].replace('-', np.nan), errors='coerce')
                    if numeric_pe.notna().any():
                        df_styled = df_styled.highlight_min(subset=['市盈率'], color='lightblue')

                st.dataframe(df_styled, use_container_width=True, height=600)

                # 下载按钮（提供多种编码选项）
                col1, col2 = st.columns(2)

                with col1:
                    # Excel兼容版本（GBK编码）
                    try:
                        csv_gbk = df_results.to_csv(index=False, encoding='gbk')
                        st.download_button(
                            "📥 下载数据(Excel版)",
                            csv_gbk,
                            f"放量突破_{datetime.now().strftime('%Y%m%d_%H%M')}_excel.csv",
                            "text/csv",
                            use_container_width=True,
                            help="适用于Windows Excel直接打开"
                        )
                    except:
                        # 如果GBK编码失败，使用GB18030（更大的字符集）
                        csv_gb = df_results.to_csv(index=False, encoding='gb18030')
                        st.download_button(
                            "📥 下载数据(Excel版)",
                            csv_gb,
                            f"放量突破_{datetime.now().strftime('%Y%m%d_%H%M')}_excel.csv",
                            "text/csv",
                            use_container_width=True,
                            help="适用于Windows Excel直接打开"
                        )

                with col2:
                    # UTF-8版本（通用版本）
                    csv_utf8 = df_results.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        "📥 下载数据(UTF-8版)",
                        csv_utf8,
                        f"放量突破_{datetime.now().strftime('%Y%m%d_%H%M')}_utf8.csv",
                        "text/csv",
                        use_container_width=True,
                        help="适用于其他软件或编程使用"
                    )
            else:
                if check_confirmation:
                    st.info("未发现符合条件且已确认突破的股票")
                    st.warning(
                        "提示：突破确认模式要求股价在放量后2-5天内站稳，筛选条件较严格。可以尝试关闭突破确认或调整其他参数。")
                else:
                    st.info("未发现符合条件的放量突破股票")

    with tab2:
        if 'scan_results' in st.session_state and not st.session_state['scan_results'].empty:
            df_results = st.session_state['scan_results']

            # 执行深度分析
            with st.spinner("正在进行深度分析..."):
                analysis = analyzer.analyze_breakout_stocks(df_results)

            # 显示分析结果
            st.markdown("### 📊 市场分析")

            # 概览指标
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("放量股票数", f"{analysis['total_count']} 只")
            with col2:
                st.metric("平均放量倍数", f"{analysis['avg_volume_ratio']:.2f}倍")
            with col3:
                st.metric("平均涨幅", f"{analysis['avg_change_pct']:.2f}%")
            with col4:
                st.metric("总成交额", f"{analysis['total_amount']:.1f}亿")

            st.markdown("---")

            # 行业分布
            st.markdown("### 🏭 行业分布")

            if 'industry_distribution' in analysis and not analysis['industry_distribution'].empty:
                industry_dist = analysis['industry_distribution']

                # 创建两列布局
                col1, col2 = st.columns(2)

                with col1:
                    # 饼图
                    fig = px.pie(
                        values=industry_dist['股票数量'],
                        names=industry_dist.index,
                        title="行业分布饼图"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # 柱状图
                    fig = px.bar(
                        x=industry_dist.index[:10],
                        y=industry_dist['平均涨幅'][:10],
                        title="Top10行业平均涨幅",
                        color=industry_dist['平均涨幅'][:10],
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # 显示详细表格
                st.dataframe(
                    industry_dist.style.highlight_max(axis=0, color='lightgreen'),
                    use_container_width=True
                )

            st.markdown("---")

            # Top榜单
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🔝 放量Top10")
                top_volume_cols = ['代码', '名称', '行业', '放量倍数', '涨幅%'] if '行业' in analysis[
                    'top_volume'].columns else ['代码', '名称', '放量倍数', '涨幅%']
                top_volume = analysis['top_volume'][top_volume_cols]
                st.dataframe(top_volume, use_container_width=True, hide_index=True)

            with col2:
                st.markdown("### 📈 涨幅Top10")
                top_change_cols = ['代码', '名称', '行业', '涨幅%', '放量倍数'] if '行业' in analysis[
                    'top_change'].columns else ['代码', '名称', '涨幅%', '放量倍数']
                top_change = analysis['top_change'][top_change_cols]
                st.dataframe(top_change, use_container_width=True, hide_index=True)

        else:
            st.info("请先进行扫描以查看分析结果")

    with tab3:
        if 'scan_results' in st.session_state and not st.session_state['scan_results'].empty:
            df_results = st.session_state['scan_results']

            # 创建热力图数据
            st.markdown("### 🔥 市场热力图")

            # 准备数据
            df_heatmap = df_results[['代码', '名称', '放量倍数', '涨幅%', '成交额(亿)']].copy()

            # 创建散点图
            fig = px.scatter(
                df_heatmap,
                x='放量倍数',
                y='涨幅%',
                size='成交额(亿)',
                color='涨幅%',
                hover_data=['代码', '名称'],
                title="放量与涨幅关系图",
                color_continuous_scale='RdYlGn',
                size_max=50
            )

            fig.update_layout(
                xaxis_title="放量倍数",
                yaxis_title="涨幅(%)",
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)

            # 成交额分布
            st.markdown("### 💰 成交额分布")

            fig = px.histogram(
                df_results,
                x='成交额(亿)',
                nbins=30,
                title="成交额分布直方图"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("请先进行扫描以查看市场热力图")

    # 自动刷新逻辑
    if auto_refresh and 'last_refresh' in st.session_state:
        time_diff = (datetime.now() - st.session_state['last_refresh']).seconds
        if time_diff >= refresh_interval * 60:
            st.rerun()

    # 页脚信息
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if 'scan_time' in st.session_state:
            st.info(f"📅 最后扫描: {st.session_state['scan_time'].strftime('%Y-%m-%d %H:%M:%S')}")

    with col2:
        st.info("📊 数据来源: Baostock")

    with col3:
        st.info("⚠️ 仅供参考，不构成投资建议")


if __name__ == "__main__":
    main()