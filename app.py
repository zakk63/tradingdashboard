import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import ta
from ta.utils import dropna
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Gold Futures Dashboard - GCZ25.CMX",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .positive {
        color: #00ff00;
    }
    .negative {
        color: #ff0000;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🥇 Gold Futures Interactive Dashboard")
st.markdown("### GCZ25.CMX - December 2025 Gold Futures Contract")

# Sidebar configuration
st.sidebar.header("⚙️ Dashboard Settings")

# Time period selection
time_periods = {
    "1 Month": "1mo",
    "3 Months": "3mo", 
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y"
}

selected_period = st.sidebar.selectbox(
    "Select Time Period:",
    list(time_periods.keys()),
    index=2
)

# Chart type selection
chart_types = ["Candlestick", "Line", "Area"]
chart_type = st.sidebar.selectbox("Chart Type:", chart_types)

# Technical indicators selection
st.sidebar.subheader("📊 Technical Indicators")
show_sma = st.sidebar.checkbox("Simple Moving Average (SMA)", value=True)
sma_period = st.sidebar.slider("SMA Period", 5, 200, 20)

show_ema = st.sidebar.checkbox("Exponential Moving Average (EMA)")
ema_period = st.sidebar.slider("EMA Period", 5, 200, 12)

show_bollinger = st.sidebar.checkbox("Bollinger Bands")
bb_period = st.sidebar.slider("BB Period", 5, 50, 20)

show_rsi = st.sidebar.checkbox("RSI", value=True)
rsi_period = st.sidebar.slider("RSI Period", 5, 50, 14)

show_macd = st.sidebar.checkbox("MACD")
show_volume = st.sidebar.checkbox("Volume", value=True)

# Data fetching function with caching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_data(symbol, period):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            st.error("No data found for the specified symbol and period.")
            return None
            
        # Calculate technical indicators
        data = dropna(data)
        
        # Moving averages
        if len(data) >= sma_period:
            data[f'SMA_{sma_period}'] = ta.trend.sma_indicator(data['Close'], window=sma_period)
        
        if len(data) >= ema_period:
            data[f'EMA_{ema_period}'] = ta.trend.ema_indicator(data['Close'], window=ema_period)
        
        # Bollinger Bands
        if len(data) >= bb_period:
            bollinger = ta.volatility.BollingerBands(data['Close'], window=bb_period)
            data['BB_High'] = bollinger.bollinger_hband()
            data['BB_Low'] = bollinger.bollinger_lband()
            data['BB_Mid'] = bollinger.bollinger_mavg()
        
        # RSI
        if len(data) >= rsi_period:
            data[f'RSI_{rsi_period}'] = ta.momentum.rsi(data['Close'], window=rsi_period)
        
        # MACD
        if len(data) >= 26:
            macd = ta.trend.MACD(data['Close'])
            data['MACD'] = macd.macd()
            data['MACD_Signal'] = macd.macd_signal()
            data['MACD_Histogram'] = macd.macd_diff()
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Fetch data
symbol = "GCZ25.CMX"
data = fetch_data(symbol, time_periods[selected_period])

if data is not None and not data.empty:
    # Current price and metrics
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
    
    # Display key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Current Price",
            f"${current_price:.2f}",
            f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col2:
        high_52w = data['High'].max()
        st.metric("Period High", f"${high_52w:.2f}")
    
    with col3:
        low_52w = data['Low'].min()
        st.metric("Period Low", f"${low_52w:.2f}")
    
    with col4:
        avg_volume = data['Volume'].mean()
        st.metric("Avg Volume", f"{avg_volume:,.0f}")
    
    with col5:
        volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
        st.metric("Volatility (Annual)", f"{volatility:.2f}%")
    
    # Main chart
    st.subheader("📈 Price Chart")
    
    # Create subplots
    rows = 1
    if show_rsi:
        rows += 1
    if show_macd:
        rows += 1
    if show_volume:
        rows += 1
    
    subplot_titles = ["Price"]
    if show_volume:
        subplot_titles.append("Volume")
    if show_rsi:
        subplot_titles.append("RSI")
    if show_macd:
        subplot_titles.append("MACD")
    
    fig = make_subplots(
        rows=rows, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
        row_heights=[0.6] + [0.4/(rows-1)]*(rows-1) if rows > 1 else [1]
    )
    
    # Price chart
    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Gold Futures",
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
    elif chart_type == "Line":
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name="Close Price",
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
    elif chart_type == "Area":
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                fill='tonexty',
                mode='lines',
                name="Close Price",
                line=dict(color='blue'),
                fillcolor='rgba(0,100,80,0.2)'
            ),
            row=1, col=1
        )
    
    # Add technical indicators
    if show_sma and f'SMA_{sma_period}' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[f'SMA_{sma_period}'],
                mode='lines',
                name=f'SMA ({sma_period})',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    if show_ema and f'EMA_{ema_period}' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[f'EMA_{ema_period}'],
                mode='lines',
                name=f'EMA ({ema_period})',
                line=dict(color='purple', width=1)
            ),
            row=1, col=1
        )
    
    if show_bollinger and 'BB_High' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_High'],
                mode='lines',
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_Low'],
                mode='lines',
                name='Bollinger Bands',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)'
            ),
            row=1, col=1
        )
    
    current_row = 1
    
    # Volume
    if show_volume:
        current_row += 1
        colors = ['red' if close < open else 'green' for close, open in zip(data['Close'], data['Open'])]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name="Volume",
                marker_color=colors,
                opacity=0.7
            ),
            row=current_row, col=1
        )
    
    # RSI
    if show_rsi and f'RSI_{rsi_period}' in data.columns:
        current_row += 1
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[f'RSI_{rsi_period}'],
                mode='lines',
                name=f'RSI ({rsi_period})',
                line=dict(color='purple', width=2)
            ),
            row=current_row, col=1
        )
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1, opacity=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1, opacity=0.5)
        fig.add_hline(y=50, line_dash="dash", line_color="gray", row=current_row, col=1, opacity=0.3)
    
    # MACD
    if show_macd and 'MACD' in data.columns:
        current_row += 1
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=2)
            ),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD_Signal'],
                mode='lines',
                name='Signal',
                line=dict(color='red', width=1)
            ),
            row=current_row, col=1
        )
        
        # MACD Histogram
        colors = ['green' if val >= 0 else 'red' for val in data['MACD_Histogram']]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['MACD_Histogram'],
                name='Histogram',
                marker_color=colors,
                opacity=0.6
            ),
            row=current_row, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"Gold Futures (GCZ25.CMX) - {selected_period}",
        xaxis_rangeslider_visible=False,
        height=600 if rows == 1 else 800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics and analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Statistical Summary")
        stats_data = {
            "Metric": ["Mean", "Median", "Std Dev", "Min", "Max", "Skewness", "Kurtosis"],
            "Value": [
                f"${data['Close'].mean():.2f}",
                f"${data['Close'].median():.2f}",
                f"${data['Close'].std():.2f}",
                f"${data['Close'].min():.2f}",
                f"${data['Close'].max():.2f}",
                f"{data['Close'].skew():.3f}",
                f"{data['Close'].kurtosis():.3f}"
            ]
        }
        st.dataframe(pd.DataFrame(stats_data), hide_index=True)
    
    with col2:
        st.subheader("🎯 Recent Performance")
        returns = data['Close'].pct_change().dropna()
        
        performance_data = {
            "Period": ["1 Day", "5 Days", "30 Days", "Overall"],
            "Return": [
                f"{returns.iloc[-1]*100:+.2f}%" if len(returns) > 0 else "N/A",
                f"{(data['Close'].iloc[-1]/data['Close'].iloc[-5]-1)*100:+.2f}%" if len(data) > 5 else "N/A",
                f"{(data['Close'].iloc[-1]/data['Close'].iloc[-30]-1)*100:+.2f}%" if len(data) > 30 else "N/A",
                f"{(data['Close'].iloc[-1]/data['Close'].iloc[0]-1)*100:+.2f}%" if len(data) > 1 else "N/A"
            ]
        }
        st.dataframe(pd.DataFrame(performance_data), hide_index=True)
    
    # Recent data table
    st.subheader("📋 Recent Data")
    recent_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10)
    recent_data = recent_data.round(2)
    st.dataframe(recent_data)
    
    # Market insights
    st.subheader("💡 Market Insights")
    
    # Calculate some basic signals
    insights = []
    
    if show_rsi and f'RSI_{rsi_period}' in data.columns:
        current_rsi = data[f'RSI_{rsi_period}'].iloc[-1]
        if current_rsi > 70:
            insights.append("🔴 RSI indicates overbought conditions")
        elif current_rsi < 30:
            insights.append("🟢 RSI indicates oversold conditions")
        else:
            insights.append(f"🟡 RSI is neutral at {current_rsi:.2f}")
    
    if show_sma and f'SMA_{sma_period}' in data.columns:
        current_sma = data[f'SMA_{sma_period}'].iloc[-1]
        if current_price > current_sma:
            insights.append(f"🟢 Price is above SMA({sma_period})")
        else:
            insights.append(f"🔴 Price is below SMA({sma_period})")
    
    # Volume analysis
    avg_volume_recent = data['Volume'].tail(5).mean()
    if data['Volume'].iloc[-1] > avg_volume_recent * 1.5:
        insights.append("📈 High volume detected - increased market interest")
    elif data['Volume'].iloc[-1] < avg_volume_recent * 0.5:
        insights.append("📉 Low volume - reduced market activity")
    
    if insights:
        for insight in insights:
            st.write(insight)
    else:
        st.write("No specific signals detected at this time.")
    
    # Footer
    st.markdown("---")
    st.markdown("*Data provided by Yahoo Finance. This dashboard is for educational purposes only and should not be used as investment advice.*")

else:
    st.error("Unable to fetch data for GCZ25.CMX. Please check if the symbol is correct and try again.")
    st.info("Note: GCZ25.CMX represents December 2025 Gold Futures. Make sure the contract is still active and trading.")

# Refresh button
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()