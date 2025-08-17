#!/usr/bin/env python3
"""
All-in-One Weekly Trading Dashboard
Author: AI Assistant
Description: Streamlit dashboard for MES, MNQ, and MGC futures analysis
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Futures Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
TICKERS = {
    'MES': 'MES=F',
    'MNQ': 'MNQ=F', 
    'MGC': 'GC=F'  # Using GC=F as GCZ25.CMX may not be available
}
SNAPSHOTS_DIR = 'snapshots'
LOOKBACK_DAYS = 90  # 3 months

def ensure_snapshots_directory():
    """Create snapshots directory if it doesn't exist"""
    if not os.path.exists(SNAPSHOTS_DIR):
        os.makedirs(SNAPSHOTS_DIR)
        st.info(f"Created {SNAPSHOTS_DIR} directory")

def fetch_futures_data(ticker_symbol, days=LOOKBACK_DAYS):
    """
    Fetch OHLCV data from Yahoo Finance
    
    Args:
        ticker_symbol (str): Yahoo Finance ticker symbol
        days (int): Number of days to look back
    
    Returns:
        pd.DataFrame: OHLCV data
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            st.error(f"No data found for {ticker_symbol}")
            return pd.DataFrame()
        
        # Clean column names and ensure we have the right columns
        data = data.round(4)
        data.index.name = 'Date'
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {str(e)}")
        return pd.DataFrame()

def calculate_sma(data, window=50):
    """Calculate Simple Moving Average"""
    if 'Close' in data.columns and len(data) >= window:
        return data['Close'].rolling(window=window).mean()
    return pd.Series(dtype=float)

def determine_trend(data):
    """
    Determine trend based on close price vs 50-day SMA
    
    Args:
        data (pd.DataFrame): OHLCV data
    
    Returns:
        str: 'Uptrend' or 'Downtrend'
    """
    if data.empty or len(data) < 50:
        return 'Insufficient Data'
    
    sma_50 = calculate_sma(data, 50)
    last_close = data['Close'].iloc[-1]
    last_sma = sma_50.iloc[-1]
    
    if pd.isna(last_sma):
        return 'Insufficient Data'
    
    return 'Uptrend' if last_close > last_sma else 'Downtrend'

def find_support_resistance(data, window=20, min_strength=2):
    """
    Find support and resistance levels using local extrema
    
    Args:
        data (pd.DataFrame): OHLCV data
        window (int): Window for finding local extrema
        min_strength (int): Minimum strength for significant levels
    
    Returns:
        dict: Dictionary with 'support' and 'resistance' levels
    """
    if data.empty or len(data) < window * 2:
        return {'support': [], 'resistance': []}
    
    highs = data['High'].values
    lows = data['Low'].values
    
    # Find local maxima (resistance) and minima (support)
    resistance_levels = []
    support_levels = []
    
    for i in range(window, len(data) - window):
        # Check for local maximum (resistance)
        if highs[i] == max(highs[i-window:i+window+1]):
            resistance_levels.append(highs[i])
        
        # Check for local minimum (support)
        if lows[i] == min(lows[i-window:i+window+1]):
            support_levels.append(lows[i])
    
    # Get the last 2 most significant levels
    resistance_levels = sorted(set(resistance_levels), reverse=True)[:2]
    support_levels = sorted(set(support_levels))[-2:]
    
    return {
        'support': support_levels,
        'resistance': resistance_levels
    }

def save_snapshot(ticker_name, ticker_symbol):
    """
    Save weekly snapshot of data and analysis
    
    Args:
        ticker_name (str): Display name of ticker
        ticker_symbol (str): Yahoo Finance symbol
    """
    # Fetch fresh data
    data = fetch_futures_data(ticker_symbol)
    
    if data.empty:
        return False
    
    # Save raw OHLCV data
    ohlcv_file = os.path.join(SNAPSHOTS_DIR, f'{ticker_name}_ohlcv.csv')
    data.to_csv(ohlcv_file)
    
    # Calculate analysis
    trend = determine_trend(data)
    sr_levels = find_support_resistance(data)
    sma_50 = calculate_sma(data, 50)
    
    # Create analysis summary
    analysis = {
        'ticker': ticker_name,
        'symbol': ticker_symbol,
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'trend': trend,
        'last_close': data['Close'].iloc[-1] if not data.empty else 0,
        'sma_50': sma_50.iloc[-1] if not sma_50.empty else 0,
        'support_levels': sr_levels['support'],
        'resistance_levels': sr_levels['resistance'],
        'total_records': len(data)
    }
    
    # Save analysis
    analysis_file = os.path.join(SNAPSHOTS_DIR, f'{ticker_name}_analysis.csv')
    analysis_df = pd.DataFrame([analysis])
    analysis_df.to_csv(analysis_file, index=False)
    
    return True

def load_snapshot(ticker_name):
    """
    Load existing snapshot data
    
    Args:
        ticker_name (str): Display name of ticker
    
    Returns:
        tuple: (ohlcv_data, analysis_data)
    """
    ohlcv_file = os.path.join(SNAPSHOTS_DIR, f'{ticker_name}_ohlcv.csv')
    analysis_file = os.path.join(SNAPSHOTS_DIR, f'{ticker_name}_analysis.csv')
    
    ohlcv_data = pd.DataFrame()
    analysis_data = {}
    
    try:
        if os.path.exists(ohlcv_file):
            ohlcv_data = pd.read_csv(ohlcv_file, index_col=0, parse_dates=True)
        
        if os.path.exists(analysis_file):
            analysis_df = pd.read_csv(analysis_file)
            if not analysis_df.empty:
                analysis_data = analysis_df.iloc[0].to_dict()
                # Convert string representations back to lists
                if 'support_levels' in analysis_data:
                    try:
                        analysis_data['support_levels'] = eval(analysis_data['support_levels'])
                    except:
                        analysis_data['support_levels'] = []
                if 'resistance_levels' in analysis_data:
                    try:
                        analysis_data['resistance_levels'] = eval(analysis_data['resistance_levels'])
                    except:
                        analysis_data['resistance_levels'] = []
    
    except Exception as e:
        st.error(f"Error loading snapshot for {ticker_name}: {str(e)}")
    
    return ohlcv_data, analysis_data

def create_candlestick_chart(data, ticker_name, analysis_data):
    """
    Create interactive candlestick chart with Plotly
    
    Args:
        data (pd.DataFrame): OHLCV data
        ticker_name (str): Display name
        analysis_data (dict): Analysis results
    
    Returns:
        plotly.graph_objects.Figure: Interactive chart
    """
    if data.empty:
        return go.Figure().add_annotation(
            text="No data available", 
            xref="paper", yref="paper",
            x=0.5, y=0.5, 
            showarrow=False,
            font=dict(size=20)
        )
    
    # Create candlestick chart
    fig = go.Figure()
    
    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=ticker_name,
        increasing_line_color='green',
        decreasing_line_color='red'
    ))
    
    # Add SMA 50
    sma_50 = calculate_sma(data, 50)
    if not sma_50.empty:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=sma_50,
            mode='lines',
            name='SMA 50',
            line=dict(color='blue', width=2)
        ))
    
    # Add support and resistance lines
    if analysis_data:
        support_levels = analysis_data.get('support_levels', [])
        resistance_levels = analysis_data.get('resistance_levels', [])
        
        for level in support_levels:
            fig.add_hline(
                y=level, 
                line_dash="dash", 
                line_color="green",
                annotation_text=f"Support: {level:.2f}",
                annotation_position="bottom right"
            )
        
        for level in resistance_levels:
            fig.add_hline(
                y=level, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Resistance: {level:.2f}",
                annotation_position="top right"
            )
    
    # Update layout
    fig.update_layout(
        title=f'{ticker_name} - Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def display_ticker_analysis(ticker_name, ticker_symbol):
    """
    Display complete analysis for a single ticker
    
    Args:
        ticker_name (str): Display name
        ticker_symbol (str): Yahoo Finance symbol
    """
    # Load existing snapshot
    ohlcv_data, analysis_data = load_snapshot(ticker_name)
    
    # Create columns for metrics and update button
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        if analysis_data:
            trend_color = "green" if analysis_data.get('trend') == 'Uptrend' else "red"
            st.metric(
                "Trend", 
                analysis_data.get('trend', 'Unknown'),
                delta=None
            )
            st.markdown(f"<span style='color: {trend_color}'>●</span>", unsafe_allow_html=True)
    
    with col2:
        if analysis_data:
            st.metric(
                "Last Close", 
                f"${analysis_data.get('last_close', 0):.2f}",
                delta=None
            )
    
    with col3:
        if analysis_data:
            st.metric(
                "SMA 50", 
                f"${analysis_data.get('sma_50', 0):.2f}",
                delta=None
            )
    
    with col4:
        if st.button(f"Update {ticker_name}", key=f"update_{ticker_name}"):
            with st.spinner(f"Updating {ticker_name} data..."):
                if save_snapshot(ticker_name, ticker_symbol):
                    st.success(f"{ticker_name} updated!")
                    st.experimental_rerun()
                else:
                    st.error(f"Failed to update {ticker_name}")
    
    # Display last update time
    if analysis_data and 'last_update' in analysis_data:
        st.caption(f"Last updated: {analysis_data['last_update']}")
    
    # Create and display chart
    if not ohlcv_data.empty:
        chart = create_candlestick_chart(ohlcv_data, ticker_name, analysis_data)
        st.plotly_chart(chart, use_container_width=True)
        
        # Display support and resistance levels
        if analysis_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Support Levels")
                support_levels = analysis_data.get('support_levels', [])
                if support_levels:
                    for i, level in enumerate(support_levels, 1):
                        st.write(f"S{i}: ${level:.2f}")
                else:
                    st.write("No support levels identified")
            
            with col2:
                st.subheader("Resistance Levels")
                resistance_levels = analysis_data.get('resistance_levels', [])
                if resistance_levels:
                    for i, level in enumerate(resistance_levels, 1):
                        st.write(f"R{i}: ${level:.2f}")
                else:
                    st.write("No resistance levels identified")
    else:
        st.warning(f"No data available for {ticker_name}. Click 'Update {ticker_name}' to fetch data.")

def main():
    """Main Streamlit application"""
    
    # Ensure snapshots directory exists
    ensure_snapshots_directory()
    
    # App header
    st.title("📈 Futures Trading Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # Check for existing snapshots
    snapshots_exist = all(
        os.path.exists(os.path.join(SNAPSHOTS_DIR, f'{ticker}_ohlcv.csv'))
        for ticker in TICKERS.keys()
    )
    
    if not snapshots_exist:
        st.sidebar.warning("⚠️ Initial setup required")
        if st.sidebar.button("🚀 Initialize All Data", key="init_all"):
            progress_bar = st.sidebar.progress(0)
            for i, (ticker_name, ticker_symbol) in enumerate(TICKERS.items()):
                st.sidebar.write(f"Fetching {ticker_name}...")
                save_snapshot(ticker_name, ticker_symbol)
                progress_bar.progress((i + 1) / len(TICKERS))
            st.sidebar.success("✅ All data initialized!")
            st.experimental_rerun()
    else:
        # Weekly update option
        st.sidebar.info("💡 Weekly Update")
        if st.sidebar.button("🔄 Update All Snapshots", key="update_all"):
            progress_bar = st.sidebar.progress(0)
            for i, (ticker_name, ticker_symbol) in enumerate(TICKERS.items()):
                st.sidebar.write(f"Updating {ticker_name}...")
                save_snapshot(ticker_name, ticker_symbol)
                progress_bar.progress((i + 1) / len(TICKERS))
            st.sidebar.success("✅ All snapshots updated!")
            st.experimental_rerun()
    
    # Display ticker selection
    selected_ticker = st.sidebar.selectbox(
        "Select Ticker to View",
        options=list(TICKERS.keys()),
        index=0
    )
    
    # Display analysis for selected ticker
    if selected_ticker:
        ticker_symbol = TICKERS[selected_ticker]
        display_ticker_analysis(selected_ticker, ticker_symbol)
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** This dashboard fetches data from Yahoo Finance and is for educational purposes only.")

if __name__ == "__main__":
    main()