#!/usr/bin/env python3
"""
Modern Swing Trading Dashboard
Advanced futures analysis with sleek UI design
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

# Configure Streamlit page with modern theme
st.set_page_config(
    page_title="Swing Trading Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern design with better contrast
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2C5AA0 0%, #1e3c72 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #2C5AA0 0%, #1e3c72 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        margin: 0.5rem 0;
    }
    
    .bullish-card {
        background: linear-gradient(135deg, #27AE60 0%, #2ECC71 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px rgba(39,174,96,0.3);
        border: 2px solid rgba(255,255,255,0.1);
    }
    
    .bearish-card {
        background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px rgba(231,76,60,0.3);
        border: 2px solid rgba(255,255,255,0.1);
    }
    
    .neutral-card {
        background: linear-gradient(135deg, #F39C12 0%, #E67E22 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px rgba(243,156,18,0.3);
        border: 2px solid rgba(255,255,255,0.1);
    }
    
    .zone-card {
        background: rgba(44,90,160,0.1);
        border: 1px solid rgba(44,90,160,0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        color: #2C5AA0;
        font-weight: 500;
    }
    
    .alert-card {
        background: linear-gradient(135deg, #E67E22 0%, #D35400 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border-left: 4px solid #F39C12;
        font-weight: 500;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 700;
        margin: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .bullish-badge {
        background: linear-gradient(135deg, #27AE60 0%, #2ECC71 100%);
        color: white;
        border: 2px solid #27AE60;
    }
    
    .bearish-badge {
        background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%);
        color: white;
        border: 2px solid #E74C3C;
    }
    
    .neutral-badge {
        background: linear-gradient(135deg, #F39C12 0%, #E67E22 100%);
        color: white;
        border: 2px solid #F39C12;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #2C5AA0 0%, #1e3c72 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(44,90,160,0.3);
        background: linear-gradient(135deg, #1e3c72 0%, #2C5AA0 100%);
    }
</style>
""", unsafe_allow_html=True)

# Constants
TICKERS = {
    'MES': {'symbol': 'MES=F', 'name': 'Micro S&P 500', 'color': '#4facfe'},
    'MNQ': {'symbol': 'MNQ=F', 'name': 'Micro Nasdaq', 'color': '#667eea'},
    'MGC': {'symbol': 'GC=F', 'name': 'Gold Futures', 'color': '#ffd700'}
}
SNAPSHOTS_DIR = 'snapshots'
LOOKBACK_DAYS = 90  # 3 months for swing analysis

def ensure_directory():
    """Create snapshots directory"""
    if not os.path.exists(SNAPSHOTS_DIR):
        os.makedirs(SNAPSHOTS_DIR)

def fetch_data(symbol, days=LOOKBACK_DAYS):
    """Fetch market data with error handling - 3 months for current structure"""
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start, end=end)
        
        if data.empty:
            return pd.DataFrame()
        
        data = data.round(4)
        return data
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

def detect_swing_zones(data, window=7, zone_tolerance=0.015):
    """Detect swing zones instead of exact points - more realistic approach"""
    if len(data) < window * 3:
        return pd.DataFrame(), pd.DataFrame()
    
    highs = data['High'].values
    lows = data['Low'].values
    dates = data.index
    
    # Find local extremes first
    potential_highs = []
    potential_lows = []
    
    # Detect potential swing highs
    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            potential_highs.append({
                'Date': pd.Timestamp(dates[i]).tz_localize(None) if hasattr(dates[i], 'tz_localize') else pd.Timestamp(dates[i]),
                'Price': float(highs[i]),
                'Index': i
            })
    
    # Detect potential swing lows
    for i in range(window, len(lows) - window):
        if lows[i] == min(lows[i-window:i+window+1]):
            potential_lows.append({
                'Date': pd.Timestamp(dates[i]).tz_localize(None) if hasattr(dates[i], 'tz_localize') else pd.Timestamp(dates[i]),
                'Price': float(lows[i]),
                'Index': i
            })
    
    # Group nearby extremes into zones
    swing_high_zones = []
    swing_low_zones = []
    
    # Process swing high zones
    if potential_highs:
        potential_highs.sort(key=lambda x: x['Date'])
        used_indices = set()
        
        for i, high in enumerate(potential_highs):
            if i in used_indices:
                continue
                
            # Find all highs within zone tolerance
            zone_highs = [high]
            zone_price = high['Price']
            
            for j, other_high in enumerate(potential_highs[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                price_diff = abs(other_high['Price'] - zone_price) / zone_price
                if price_diff <= zone_tolerance:
                    zone_highs.append(other_high)
                    used_indices.add(j)
                    # Update zone price to average
                    zone_price = sum(h['Price'] for h in zone_highs) / len(zone_highs)
            
            used_indices.add(i)
            
            # Create zone if significant
            if len(zone_highs) >= 1:  # Even single touches can be significant
                swing_high_zones.append({
                    'Date': max(h['Date'] for h in zone_highs),  # Most recent touch
                    'Price': zone_price,
                    'Type': 'High Zone',
                    'Touches': len(zone_highs),
                    'First_Touch': min(h['Date'] for h in zone_highs),
                    'Last_Touch': max(h['Date'] for h in zone_highs)
                })
    
    # Process swing low zones
    if potential_lows:
        potential_lows.sort(key=lambda x: x['Date'])
        used_indices = set()
        
        for i, low in enumerate(potential_lows):
            if i in used_indices:
                continue
                
            # Find all lows within zone tolerance
            zone_lows = [low]
            zone_price = low['Price']
            
            for j, other_low in enumerate(potential_lows[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                price_diff = abs(other_low['Price'] - zone_price) / zone_price
                if price_diff <= zone_tolerance:
                    zone_lows.append(other_low)
                    used_indices.add(j)
                    # Update zone price to average
                    zone_price = sum(l['Price'] for l in zone_lows) / len(zone_lows)
            
            used_indices.add(i)
            
            # Create zone if significant
            if len(zone_lows) >= 1:  # Even single touches can be significant
                swing_low_zones.append({
                    'Date': max(l['Date'] for l in zone_lows),  # Most recent touch
                    'Price': zone_price,
                    'Type': 'Low Zone',
                    'Touches': len(zone_lows),
                    'First_Touch': min(l['Date'] for l in zone_lows),
                    'Last_Touch': max(l['Date'] for l in zone_lows)
                })
    
    return pd.DataFrame(swing_high_zones), pd.DataFrame(swing_low_zones)

def analyze_structure(swing_highs_df, swing_lows_df):
    """Analyze market structure using swing zones - much more realistic"""
    if swing_highs_df.empty or swing_lows_df.empty:
        return {
            'trend': 'Insufficient Data',
            'bias': 'Neutral',
            'confidence': 0,
            'details': 'Need swing zones for analysis',
            'last_highs': [],
            'last_lows': []
        }
    
    # Sort zones by date
    high_zones = swing_highs_df.sort_values('Date')
    low_zones = swing_lows_df.sort_values('Date')
    
    if len(high_zones) < 2 or len(low_zones) < 2:
        return {
            'trend': 'Choppy',
            'bias': 'Neutral',
            'confidence': 25,
            'details': f'Insufficient zones: {len(high_zones)} high zones, {len(low_zones)} low zones',
            'last_highs': high_zones.to_dict('records'),
            'last_lows': low_zones.to_dict('records')
        }
    
    # Get recent zones (focus on last 3-4 of each)
    recent_high_zones = high_zones.tail(4)
    recent_low_zones = low_zones.tail(4)
    
    # Analyze zone progression for structure
    structure_signals = {
        'higher_highs': 0,
        'lower_highs': 0,
        'higher_lows': 0,
        'lower_lows': 0,
        'consolidation': 0
    }
    
    # Analyze high zone progression
    high_prices = recent_high_zones['Price'].tolist()
    for i in range(1, len(high_prices)):
        current_high = high_prices[i]
        previous_high = high_prices[i-1]
        
        price_diff_pct = (current_high - previous_high) / previous_high
        
        if price_diff_pct > 0.005:  # >0.5% higher = Higher High
            structure_signals['higher_highs'] += 1
        elif price_diff_pct < -0.005:  # <-0.5% lower = Lower High
            structure_signals['lower_highs'] += 1
        else:  # Within 0.5% = Consolidation
            structure_signals['consolidation'] += 0.5
    
    # Analyze low zone progression  
    low_prices = recent_low_zones['Price'].tolist()
    for i in range(1, len(low_prices)):
        current_low = low_prices[i]
        previous_low = low_prices[i-1]
        
        price_diff_pct = (current_low - previous_low) / previous_low
        
        if price_diff_pct > 0.005:  # >0.5% higher = Higher Low
            structure_signals['higher_lows'] += 1
        elif price_diff_pct < -0.005:  # <-0.5% lower = Lower Low
            structure_signals['lower_lows'] += 1
        else:  # Within 0.5% = Consolidation
            structure_signals['consolidation'] += 0.5
    
    # Calculate trend strength
    bullish_strength = structure_signals['higher_highs'] + structure_signals['higher_lows']
    bearish_strength = structure_signals['lower_highs'] + structure_signals['lower_lows']
    consolidation_strength = structure_signals['consolidation']
    
    # Determine overall structure
    total_signals = bullish_strength + bearish_strength + consolidation_strength
    
    if total_signals == 0:
        return {
            'trend': 'Insufficient Data',
            'bias': 'Neutral',
            'confidence': 0,
            'details': 'No clear zone progression detected',
            'last_highs': recent_high_zones.tail(2).to_dict('records'),
            'last_lows': recent_low_zones.tail(2).to_dict('records')
        }
    
    # Calculate percentages
    bullish_pct = (bullish_strength / total_signals) * 100
    bearish_pct = (bearish_strength / total_signals) * 100
    consolidation_pct = (consolidation_strength / total_signals) * 100
    
    # Determine trend with realistic thresholds
    if bullish_strength >= 2 and bullish_pct > 60:
        trend = 'Bullish'
        bias = 'Bullish'
        confidence = min(90, int(50 + bullish_pct * 0.6))
        pattern_type = "Strong HH+HL" if bullish_pct > 75 else "Moderate HH+HL"
        details = f"{pattern_type} zone pattern ({bullish_strength} bullish signals, {int(bullish_pct)}%)"
        
    elif bearish_strength >= 2 and bearish_pct > 60:
        trend = 'Bearish'
        bias = 'Bearish'
        confidence = min(90, int(50 + bearish_pct * 0.6))
        pattern_type = "Strong LH+LL" if bearish_pct > 75 else "Moderate LH+LL"
        details = f"{pattern_type} zone pattern ({bearish_strength} bearish signals, {int(bearish_pct)}%)"
        
    elif consolidation_pct > 50 or (bullish_strength == bearish_strength and bullish_strength > 0):
        trend = 'Consolidation'
        bias = 'Neutral'
        confidence = min(80, int(40 + consolidation_pct * 0.5))
        details = f"Sideways/ranging zones (Bull: {bullish_strength}, Bear: {bearish_strength}, Consol: {consolidation_strength:.1f})"
        
    else:
        trend = 'Choppy'
        bias = 'Neutral'
        confidence = 35
        details = f"Mixed zone signals (Bull: {bullish_strength}, Bear: {bearish_strength}, Consol: {consolidation_strength:.1f})"
    
    return {
        'trend': trend,
        'bias': bias,
        'confidence': confidence,
        'details': details,
        'last_highs': recent_high_zones.tail(2).to_dict('records'),
        'last_lows': recent_low_zones.tail(2).to_dict('records'),
        'total_swings': f"{len(high_zones)} high zones, {len(low_zones)} low zones detected",
        'zone_analysis': {
            'bullish_signals': bullish_strength,
            'bearish_signals': bearish_strength,
            'consolidation_signals': consolidation_strength,
            'bullish_percentage': round(bullish_pct, 1),
            'bearish_percentage': round(bearish_pct, 1)
        }
    }

def create_zones(swing_highs_df, swing_lows_df, tolerance=0.02):
    """Create support and resistance zones with clustering"""
    zones = {'support': [], 'resistance': []}
    
    # Process resistance zones from swing highs
    if not swing_highs_df.empty:
        prices = swing_highs_df['Price'].values
        dates = swing_highs_df['Date'].values
        
        resistance_levels = []
        for i, price in enumerate(prices):
            # Find nearby prices within tolerance
            nearby = np.abs(prices - price) / price <= tolerance
            if np.sum(nearby) >= 2:  # At least 2 touches
                avg_price = np.mean(prices[nearby])
                last_touch = max(dates[nearby])
                touch_count = np.sum(nearby)
                
                # Convert timestamp to handle timezone issues
                if hasattr(last_touch, 'tz_localize'):
                    last_touch = last_touch.tz_localize(None) if last_touch.tz is not None else last_touch
                
                # Calculate days ago properly
                now = pd.Timestamp.now()
                days_diff = (now - pd.Timestamp(last_touch)).days
                
                resistance_levels.append({
                    'price': avg_price,
                    'touches': touch_count,
                    'last_touch': last_touch,
                    'strength': touch_count * (1 + max(0, (365 - days_diff) / 365))
                })
        
        # Remove duplicates and sort by strength
        seen_prices = set()
        unique_levels = []
        for level in resistance_levels:
            price_key = round(level['price'], 1)
            if price_key not in seen_prices:
                seen_prices.add(price_key)
                unique_levels.append(level)
        
        zones['resistance'] = sorted(unique_levels, key=lambda x: x['strength'], reverse=True)[:3]
    
    # Process support zones from swing lows
    if not swing_lows_df.empty:
        prices = swing_lows_df['Price'].values
        dates = swing_lows_df['Date'].values
        
        support_levels = []
        for i, price in enumerate(prices):
            # Find nearby prices within tolerance
            nearby = np.abs(prices - price) / price <= tolerance
            if np.sum(nearby) >= 2:  # At least 2 touches
                avg_price = np.mean(prices[nearby])
                last_touch = max(dates[nearby])
                touch_count = np.sum(nearby)
                
                # Convert timestamp to handle timezone issues
                if hasattr(last_touch, 'tz_localize'):
                    last_touch = last_touch.tz_localize(None) if last_touch.tz is not None else last_touch
                
                # Calculate days ago properly
                now = pd.Timestamp.now()
                days_diff = (now - pd.Timestamp(last_touch)).days
                
                support_levels.append({
                    'price': avg_price,
                    'touches': touch_count,
                    'last_touch': last_touch,
                    'strength': touch_count * (1 + max(0, (365 - days_diff) / 365))
                })
        
        # Remove duplicates and sort by strength
        seen_prices = set()
        unique_levels = []
        for level in support_levels:
            price_key = round(level['price'], 1)
            if price_key not in seen_prices:
                seen_prices.add(price_key)
                unique_levels.append(level)
        
        zones['support'] = sorted(unique_levels, key=lambda x: x['strength'], reverse=True)[:3]
    
    return zones

def generate_alerts(data, structure, zones, ticker_name):
    """Generate intelligent trading alerts"""
    alerts = []
    
    if data.empty:
        return alerts
    
    current_price = data['Close'].iloc[-1]
    
    # Structure alerts
    trend = structure.get('trend', 'Unknown')
    confidence = structure.get('confidence', 0)
    
    if confidence > 70:
        if trend == 'Bullish':
            alerts.append(f"🚀 {ticker_name}: Strong bullish structure confirmed ({confidence}% confidence)")
        elif trend == 'Bearish':
            alerts.append(f"📉 {ticker_name}: Strong bearish structure confirmed ({confidence}% confidence)")
    
    # Zone proximity alerts
    for zone in zones.get('support', []):
        distance = abs(current_price - zone['price']) / current_price
        if distance <= 0.015:  # Within 1.5%
            alerts.append(f"🔔 {ticker_name}: Approaching support at ${zone['price']:.2f} ({distance*100:.1f}% away)")
    
    for zone in zones.get('resistance', []):
        distance = abs(current_price - zone['price']) / current_price
        if distance <= 0.015:  # Within 1.5%
            alerts.append(f"⚠️ {ticker_name}: Approaching resistance at ${zone['price']:.2f} ({distance*100:.1f}% away)")
    
    return alerts

def create_modern_chart(data, swing_highs, swing_lows, zones, ticker_name, color):
    """Create beautiful modern chart with dark theme"""
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, xref="paper", yref="paper")
        return fig
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.8, 0.2],
        subplot_titles=['Price Action', 'Volume']
    )
    
    # Candlestick chart with better colors
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=ticker_name,
            increasing_line_color='#27AE60',  # Strong green
            decreasing_line_color='#E74C3C',  # Strong red
            increasing_fillcolor='rgba(39,174,96,0.7)',
            decreasing_fillcolor='rgba(231,76,60,0.7)'
        ),
        row=1, col=1
    )
    
    # Add swing zones with better visualization
    if not swing_highs.empty:
        # Color code zones by number of touches
        colors = ['#E74C3C' if touches >= 2 else '#F39C12' for touches in swing_highs.get('Touches', [1] * len(swing_highs))]
        sizes = [16 if touches >= 2 else 12 for touches in swing_highs.get('Touches', [1] * len(swing_highs))]
        
        fig.add_trace(
            go.Scatter(
                x=swing_highs['Date'],
                y=swing_highs['Price'],
                mode='markers',
                name='High Zones',
                marker=dict(
                    symbol='triangle-down',
                    size=sizes,
                    color=colors,
                    line=dict(width=3, color='white'),
                    opacity=0.9
                ),
                text=[f"Zone: ${price:.2f}<br>Touches: {touches}" 
                      for price, touches in zip(swing_highs['Price'], swing_highs.get('Touches', [1] * len(swing_highs)))],
                hovertemplate='%{text}<extra></extra>'
            ),
            row=1, col=1
        )
    
    if not swing_lows.empty:
        # Color code zones by number of touches
        colors = ['#27AE60' if touches >= 2 else '#F39C12' for touches in swing_lows.get('Touches', [1] * len(swing_lows))]
        sizes = [16 if touches >= 2 else 12 for touches in swing_lows.get('Touches', [1] * len(swing_lows))]
        
        fig.add_trace(
            go.Scatter(
                x=swing_lows['Date'],
                y=swing_lows['Price'],
                mode='markers',
                name='Low Zones',
                marker=dict(
                    symbol='triangle-up',
                    size=sizes,
                    color=colors,
                    line=dict(width=3, color='white'),
                    opacity=0.9
                ),
                text=[f"Zone: ${price:.2f}<br>Touches: {touches}" 
                      for price, touches in zip(swing_lows['Price'], swing_lows.get('Touches', [1] * len(swing_lows)))],
                hovertemplate='%{text}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Add support/resistance zones with better colors
    for i, zone in enumerate(zones.get('support', [])):
        fig.add_hline(
            y=zone['price'],
            line=dict(color='#27AE60', width=3, dash='dash'),
            annotation=dict(
                text=f"S{i+1}: ${zone['price']:.2f}",
                bgcolor="rgba(39,174,96,0.9)",
                bordercolor="white",
                borderwidth=2,
                font=dict(color="white", size=12)
            ),
            row=1, col=1
        )
    
    for i, zone in enumerate(zones.get('resistance', [])):
        fig.add_hline(
            y=zone['price'],
            line=dict(color='#E74C3C', width=3, dash='dash'),
            annotation=dict(
                text=f"R{i+1}: ${zone['price']:.2f}",
                bgcolor="rgba(231,76,60,0.9)",
                bordercolor="white",
                borderwidth=2,
                font=dict(color="white", size=12)
            ),
            row=1, col=1
        )
    
    # Volume chart with better colors
    colors = ['#27AE60' if close >= open else '#E74C3C' 
              for close, open in zip(data['Close'], data['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.8
        ),
        row=2, col=1
    )
    
    # Modern dark theme layout
    fig.update_layout(
        title=dict(
            text=f'{ticker_name} - Swing Analysis',
            font=dict(size=24, color='white'),
            x=0.5
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="white",
            borderwidth=1
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # Update axes
    fig.update_xaxes(
        gridcolor='rgba(255,255,255,0.1)',
        showgrid=True,
        zeroline=False
    )
    fig.update_yaxes(
        gridcolor='rgba(255,255,255,0.1)',
        showgrid=True,
        zeroline=False
    )
    
    return fig

def save_analysis(ticker_name, ticker_symbol):
    """Save analysis data using zone-based approach"""
    data = fetch_data(ticker_symbol)
    if data.empty:
        return False
    
    # Perform zone-based analysis
    swing_highs, swing_lows = detect_swing_zones(data)
    structure = analyze_structure(swing_highs, swing_lows)
    zones = create_zones(swing_highs, swing_lows)
    alerts = generate_alerts(data, structure, zones, ticker_name)
    
    # Save to files
    ensure_directory()
    
    # Save OHLCV data
    data.to_csv(f'{SNAPSHOTS_DIR}/{ticker_name}_data.csv')
    
    # Save swing zones
    if not swing_highs.empty:
        swing_highs.to_csv(f'{SNAPSHOTS_DIR}/{ticker_name}_highs.csv', index=False)
    if not swing_lows.empty:
        swing_lows.to_csv(f'{SNAPSHOTS_DIR}/{ticker_name}_lows.csv', index=False)
    
    # Save analysis results
    analysis = {
        'ticker': ticker_name,
        'symbol': ticker_symbol,
        'last_update': datetime.now().isoformat(),
        'current_price': float(data['Close'].iloc[-1]),
        'structure': structure,
        'zones': zones,
        'alerts': alerts,
        'data_points': len(data)
    }
    
    # Convert to JSON-serializable format
    import json
    with open(f'{SNAPSHOTS_DIR}/{ticker_name}_analysis.json', 'w') as f:
        json.dump(analysis, f, default=str)
    
    return True

def load_analysis(ticker_name):
    """Load saved analysis"""
    try:
        # Load data
        data_file = f'{SNAPSHOTS_DIR}/{ticker_name}_data.csv'
        highs_file = f'{SNAPSHOTS_DIR}/{ticker_name}_highs.csv'
        lows_file = f'{SNAPSHOTS_DIR}/{ticker_name}_lows.csv'
        analysis_file = f'{SNAPSHOTS_DIR}/{ticker_name}_analysis.json'
        
        data = pd.DataFrame()
        swing_highs = pd.DataFrame()
        swing_lows = pd.DataFrame()
        analysis = {}
        
        if os.path.exists(data_file):
            data = pd.read_csv(data_file, index_col=0, parse_dates=True)
        
        if os.path.exists(highs_file):
            swing_highs = pd.read_csv(highs_file, parse_dates=['Date'])
        
        if os.path.exists(lows_file):
            swing_lows = pd.read_csv(lows_file, parse_dates=['Date'])
        
        if os.path.exists(analysis_file):
            import json
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
        
        return data, swing_highs, swing_lows, analysis
    
    except Exception as e:
        st.error(f"Error loading {ticker_name}: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

def render_ticker_card(ticker_name, ticker_info, analysis):
    """Render modern ticker card with better contrast"""
    structure = analysis.get('structure', {})
    trend = structure.get('trend', 'Unknown')
    confidence = structure.get('confidence', 0)
    current_price = analysis.get('current_price', 0)
    
    # Determine card style with better colors
    if trend == 'Bullish':
        card_class = 'bullish-card'
        badge_class = 'bullish-badge'
        emoji = '🚀'
    elif trend == 'Bearish':
        card_class = 'bearish-card'
        badge_class = 'bearish-badge'
        emoji = '📉'
    else:
        card_class = 'neutral-card'
        badge_class = 'neutral-badge'
        emoji = '🔄'
    
    st.markdown(f"""
    <div class="{card_class}">
        <h3>{emoji} {ticker_info['name']}</h3>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin: 0; font-size: 2rem;">${current_price:.2f}</h2>
                <span class="{badge_class} status-badge">{trend} {confidence}%</span>
            </div>
            <div style="text-align: right;">
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">{structure.get('details', 'No analysis')}</p>
                <p style="margin: 0; font-size: 0.8rem; opacity: 0.7;">{structure.get('total_swings', '')}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application"""
    ensure_directory()
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Swing Trading Pro</h1>', unsafe_allow_html=True)
    
    # Navigation
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    selected_ticker = None
    with col1:
        if st.button("📈 MES", use_container_width=True):
            selected_ticker = 'MES'
    with col2:
        if st.button("💻 MNQ", use_container_width=True):
            selected_ticker = 'MNQ'
    with col3:
        if st.button("🥇 MGC", use_container_width=True):
            selected_ticker = 'MGC'
    with col4:
        if st.button("🔄 Update All", use_container_width=True):
            with st.spinner("Updating all data..."):
                progress = st.progress(0)
                for i, (name, info) in enumerate(TICKERS.items()):
                    save_analysis(name, info['symbol'])
                    progress.progress((i + 1) / len(TICKERS))
            st.success("All data updated!")
            st.rerun()
    
    # Default to MES if none selected
    if selected_ticker is None:
        selected_ticker = 'MES'
    
    # Load analysis for selected ticker
    data, swing_highs, swing_lows, analysis = load_analysis(selected_ticker)
    
    # Initialize if no data
    if data.empty:
        if st.button(f"🚀 Initialize {selected_ticker} Data", use_container_width=True):
            with st.spinner(f"Analyzing {selected_ticker}..."):
                save_analysis(selected_ticker, TICKERS[selected_ticker]['symbol'])
            st.success(f"{selected_ticker} initialized!")
            st.rerun()
        return
    
    # Display ticker overview
    ticker_info = TICKERS[selected_ticker]
    render_ticker_card(selected_ticker, ticker_info, analysis)
    
    # Display alerts
    alerts = analysis.get('alerts', [])
    if alerts:
        st.markdown("### 🚨 Trading Alerts")
        for alert in alerts:
            st.markdown(f'<div class="alert-card">{alert}</div>', unsafe_allow_html=True)
    
    # Chart
    if not data.empty:
        zones = analysis.get('zones', {})
        chart = create_modern_chart(
            data, swing_highs, swing_lows, zones, 
            ticker_info['name'], ticker_info['color']
        )
        st.plotly_chart(chart, use_container_width=True)
    
    # Support/Resistance Analysis
    zones = analysis.get('zones', {})
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🟢 Support Zones")
        support_zones = zones.get('support', [])
        if support_zones:
            for i, zone in enumerate(support_zones):
                days_ago = (datetime.now() - pd.to_datetime(zone['last_touch'])).days
                st.markdown(f"""
                <div class="zone-card">
                    <strong>S{i+1}: ${zone['price']:.2f}</strong><br>
                    Touches: {zone['touches']} | Last: {days_ago} days ago
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant support zones detected")
    
    with col2:
        st.markdown("### 🔴 Resistance Zones")
        resistance_zones = zones.get('resistance', [])
        if resistance_zones:
            for i, zone in enumerate(resistance_zones):
                days_ago = (datetime.now() - pd.to_datetime(zone['last_touch'])).days
                st.markdown(f"""
                <div class="zone-card">
                    <strong>R{i+1}: ${zone['price']:.2f}</strong><br>
                    Touches: {zone['touches']} | Last: {days_ago} days ago
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant resistance zones detected")
    
    # Technical Summary
    structure = analysis.get('structure', {})
    st.markdown("### 📊 3-Month Structure Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Market Structure", structure.get('trend', 'Unknown'))
    with col2:
        st.metric("Confidence", f"{structure.get('confidence', 0)}%")
    with col3:
        total_swings = structure.get('total_swings', 'No data')
        st.metric("Swing Data", total_swings)
    
    # Show structure details
    details = structure.get('details', 'No analysis available')
    st.info(f"**Analysis:** {details}")
    
    col1, col2 = st.columns(2)
    with col1:
        last_update = analysis.get('last_update', 'Never')
        if last_update != 'Never':
            last_update = pd.to_datetime(last_update).strftime('%Y-%m-%d %H:%M')
        st.metric("Last Update", last_update)
    with col2:
        data_points = analysis.get('data_points', 0)
        st.metric("Data Points", f"{data_points} days")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Disclaimer:** This dashboard is for educational purposes only. "
        "Always perform your own analysis before making trading decisions."
    )

if __name__ == "__main__":
    main()