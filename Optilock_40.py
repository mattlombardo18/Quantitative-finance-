#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:49:45 2025

@author: mathisdelooze
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from scipy.stats import norm
from scipy.interpolate import RegularGridInterpolator

# =============================================================================
# Streamlit Interface Setup
# =============================================================================
st.set_page_config(page_title="OptiLock 40 - Monte Carlo Pricing", layout="wide")

st.markdown("""
    <h1 style='text-align: center;'>🔒 OptiLock 40 - Monte Carlo Pricing Dashboard</h1>
    <h4 style='text-align: center; color:gray;'>Combining Capital Security with Market Growth Potential</h4>
    <hr style='border: 1px solid #555;'>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='text-align: right; font-size: 14px; color: gray;'>"
    "Projet créé par : Mathis Delooze, Matthieu Lombardo et Youri Leconte</div>",
    unsafe_allow_html=True
)
# =============================================================================
# Sidebar for User Inputs
# =============================================================================
with st.sidebar:
    st.header("📊 Market Parameters")
    st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

    with st.expander("🔹 Core Market Variables", expanded=True):
        S0 = st.number_input("Initial CAC 40 Level", value=8000, min_value=1000, max_value=20000)
        r = st.number_input("Risk-Free Rate (%)", value=4.55, min_value=0.0, max_value=10.0) / 100
        div = st.number_input("Dividend Level(%)", value=3.0, min_value=0.0, max_value=10.0)/100
        T = st.number_input("Maturity (Years)", value=3, min_value=1, max_value=10)
        NSimul = st.number_input("Number of Simulations", value=10000, min_value=1000, max_value=500000, step=10000)

    with st.expander("🔹 Barrier & Participation Settings", expanded=True):
        barrier_10_pct = st.slider("Barrier Level 1 (%)", min_value=100, max_value=200, value=120, step=1)
        barrier_25_pct = st.slider("Barrier Level 2 (%)", min_value=100, max_value=300, value=150, step=1)

        Barrier_10 = (barrier_10_pct / 100) * S0
        Barrier_25 = (barrier_25_pct / 100) * S0

        st.markdown(f"""
            <div style="background-color:#f0f0f0; padding:10px; border-radius:10px; text-align: left;">
            🔹 <b>Barrier Level 1:</b> {Barrier_10:,.2f} <br>
            🔹 <b>Barrier Level 2:</b> {Barrier_25:,.2f}
            </div>
        """, unsafe_allow_html=True)

        Participation = 0.5

# =============================================================================
# Option Breakdown Table
# =============================================================================
st.markdown("<h3> Option Composition of OptiLock 40</h3>", unsafe_allow_html=True)

options_table = pd.DataFrame({
    "Option Type": ["Vanilla Call Option", "One-Touch Digital Option", "One-Touch Digital Option", "Knock-Out Call Option", "Knock-Out Call Option"],
    "Strike (K)": ["150%", "120%", "150%", "100%", "120%"],
    "Barrier (if applicable)": ["N/A", "120%", "150%", "KO at 120%", "KO at 150%"],
    "Purpose": [
        "Captures 50% of CAC 40 upside",
        "Pays 10% if CAC 40 hits 120%",
        "Pays 25% if CAC 40 hits 150%",
        "Deactivates above 120%",
        "Deactivates above 150%"
    ]
})

styled_table = options_table.style.set_properties(**{
    "text-align": "center",
    "background-color": "#f0f0f0",
    "border": "1px solid #ddd"
}).set_table_styles([
    {'selector': 'thead th', 'props': [('background-color', '#333'), ('color', 'white'), ('text-align', 'center')]}
])

st.dataframe(styled_table, width=800)

st.markdown("---")

# =============================================================================
# Volatility Surface
# =============================================================================

strike_percentages = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200])
maturities = np.array([1/12, 3/12, 6/12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # In years

# Volatility matrix
vol_matrix = np.array([
    [68.7, 53.6, 46.0, 38.9, 34.4, 32.6, 32.3, 32.0, 31.8, 31.7, 31.6, 31.6, 31.6],
    [60.5, 48.4, 42.2, 36.4, 32.8, 31.5, 31.3, 31.2, 31.1, 31.1, 31.1, 31.2, 31.1],
    [52.4, 43.2, 38.4, 33.9, 31.1, 30.2, 30.3, 30.3, 30.4, 30.4, 30.5, 30.6, 30.6],
    [44.3, 37.9, 34.5, 31.3, 29.3, 28.9, 29.2, 29.3, 29.5, 29.6, 29.8, 30.0, 30.1],
    [36.2, 32.5, 30.6, 28.6, 27.5, 27.5, 28.0, 28.3, 28.6, 28.8, 29.0, 29.2, 29.4],
    [28.0, 27.2, 26.6, 25.9, 25.7, 26.1, 26.8, 27.2, 27.5, 27.8, 28.1, 28.4, 28.6],
    [21.1, 22.8, 23.3, 23.5, 24.0, 24.8, 25.6, 26.2, 26.6, 26.9, 27.3, 27.6, 27.9],
    [16.1, 19.6, 20.8, 21.7, 22.7, 23.7, 24.7, 25.3, 25.8, 26.2, 26.6, 27.0, 27.2],
    [15.9, 17.3, 18.9, 20.3, 21.7, 22.9, 23.9, 24.6, 25.1, 25.5, 25.9, 26.4, 26.7],
    [15.6, 15.8, 17.6, 19.3, 20.8, 22.1, 23.2, 23.9, 24.5, 25.0, 25.4, 25.9, 26.2],
    [15.4, 14.9, 16.7, 18.5, 20.2, 21.6, 22.7, 23.4, 24.0, 24.5, 24.9, 25.4, 25.7],
    [15.1, 14.5, 16.2, 18.0, 19.7, 21.1, 22.2, 22.9, 23.5, 24.1, 24.5, 25.0, 25.3],
    [14.9, 14.5, 16.0, 17.7, 19.3, 20.7, 21.8, 22.5, 23.2, 23.7, 24.1, 24.6, 25.0],
    [14.6, 14.8, 16.0, 17.6, 19.0, 20.4, 21.4, 22.2, 22.8, 23.3, 23.8, 24.3, 24.6],
    [14.4, 15.3, 16.2, 17.6, 18.8, 20.1, 21.1, 21.9, 22.5, 23.0, 23.5, 24.0, 24.3],
    [14.1, 15.9, 16.5, 17.7, 18.7, 19.9, 20.9, 21.6, 22.2, 22.7, 23.2, 23.7, 24.1]
]) / 100

# Create an interpolator function
vol_interpolator = RegularGridInterpolator((strike_percentages, maturities), vol_matrix, method='linear')

strike_mesh, maturity_mesh = np.meshgrid(strike_percentages, maturities, indexing='ij')
vol_surface_values = vol_interpolator((strike_mesh, maturity_mesh))

fig = go.Figure(data=[go.Surface(
    z=vol_surface_values, 
    x=strike_mesh, 
    y=maturity_mesh, 
    colorscale='Viridis'
)])

fig.update_layout(
    scene=dict(
        xaxis_title="Strike (%)",
        yaxis_title="Maturity (Years)",
        zaxis_title="Implied Volatility"
    ),
    width=900,
    height=600
)

st.markdown("<h3> Volatility Surface</h3>", unsafe_allow_html=True)
st.plotly_chart(fig)

st.markdown("The pricing of this product is entirely based on the volatility surface. Prices are dynamically linked to it, adjusting automatically according to the user's choice of strikes, maturity, and barriers.")
st.markdown("---")

# Define volatilities for each product based on the volatility surface
vol_vanilla_call = vol_interpolator([150, T])[0]  # volatility at 150% strike
vol_digital_10 = vol_interpolator([barrier_10_pct, T])[0]  # Volatility at 120% strike
vol_digital_25 = vol_interpolator([barrier_25_pct, T])[0]  # Volatility at 150% strike
vol_ko_120 = vol_interpolator([100, T])[0]  # Volatility at 100% strike
vol_ko_150 = vol_interpolator([120, T])[0]  # Volatility at 120% strike

# =============================================================================
# Monte Carlo Simulation for CAC 40 Price Paths
# =============================================================================

# Set random seed for reproducibility
np.random.seed(42)

# Monte Carlo Parameters
N_steps = 252 * T  # Daily steps (assuming 252 trading days per year)
dt = T / N_steps  # Time step

# Initialize paths
paths_digital_10 = np.zeros((NSimul, N_steps + 1))
paths_digital_25 = np.zeros((NSimul, N_steps + 1))
paths_ko_120 = np.zeros((NSimul, N_steps + 1))
paths_ko_150 = np.zeros((NSimul, N_steps + 1))
paths_vanilla_call = np.zeros((NSimul, N_steps + 1))

# Set initial price for all paths
paths_digital_10[:, 0] = S0
paths_digital_25[:, 0] = S0
paths_ko_120[:, 0] = S0
paths_ko_150[:, 0] = S0
paths_vanilla_call[:, 0] = S0


# Generate standard normal shocks
Z = np.random.standard_normal((NSimul, N_steps))

# Function to simulate paths with specific volatility
def simulate_paths(paths, r, vol, div):
    dW = vol * np.sqrt(dt) * Z
    drift = (r - div - 0.5 * vol**2) * dt
    log_paths = np.cumsum(drift + dW, axis=1)
    paths[:, 1:] = S0 * np.exp(log_paths)

simulate_paths(paths_digital_10, r, vol_digital_10, div)
simulate_paths(paths_digital_25, r, vol_digital_25, div)
simulate_paths(paths_ko_120, r, vol_ko_120, div)
simulate_paths(paths_ko_150, r, vol_ko_150, div)
simulate_paths(paths_vanilla_call, r, vol_vanilla_call, div)

# Compute max path to check barriers (One-Touch Condition)
max_paths_digital_10 = np.max(paths_digital_10, axis=1)
max_paths_digital_25 = np.max(paths_digital_25, axis=1)

hit_10 = (max_paths_digital_10 >= Barrier_10).astype(int)  
hit_25 = (max_paths_digital_25 >= Barrier_25).astype(int)

# =============================================================================
# Payoff Computation for Each Option
# =============================================================================
final_prices_ko_120 = paths_ko_120[:, -1]
final_prices_ko_150 = paths_ko_150[:, -1]
final_prices_vanilla_call = paths_vanilla_call[:, -1]

# One-Touch Digital Options
digital_10_mc = np.where(hit_10 == 1, 10, 0)  # Pays 10% if CAC 40 touches 120%
digital_25_mc = np.where(hit_25 == 1, 15, 0)  # Pays 15% if CAC 40 touches 150%

## Knock-Out Call Options
call_ko_120_mc = np.where(hit_10 == 0,
                           np.maximum(Participation * (final_prices_ko_120 - S0) / S0, 0) * 100, 0)

call_ko_150_mc = np.where((hit_10 == 1) & (hit_25 == 0),
                           np.maximum(Participation * (final_prices_ko_150 - 1.2 * S0) / S0, 0) * 100, 0)

# Vanilla Call
vanilla_call_mc = np.maximum((final_prices_vanilla_call - 1.5 * S0) / S0, 0) * 100

# =============================================================================
# Compute Expected Payoff and Discount Factors
# =============================================================================
expected_payoff_digital_10 = np.mean(digital_10_mc)
expected_payoff_digital_25 = np.mean(digital_25_mc)
expected_payoff_call_ko_120 = np.mean(call_ko_120_mc)
expected_payoff_call_ko_150 = np.mean(call_ko_150_mc)
expected_payoff_vanilla_call = np.mean(vanilla_call_mc)

# Compute Discount Factor
discount_factor = np.exp(-r * T)

# Discounted Option Prices
price_digital_10 = expected_payoff_digital_10 * discount_factor
price_digital_25 = expected_payoff_digital_25 * discount_factor
price_call_ko_120 = expected_payoff_call_ko_120 * discount_factor  * Participation
price_call_ko_150 = expected_payoff_call_ko_150 * discount_factor  * Participation
price_vanilla_call = expected_payoff_vanilla_call * discount_factor  * Participation

# =============================================================================
# Final Structured Product Pricing
# =============================================================================
product_price = (
    price_call_ko_120 +
    price_call_ko_150 +
    price_digital_10 +
    price_digital_25 +
    price_vanilla_call
)

# =============================================================================
# Zero Coupon Bond
# =============================================================================
zero_coupon_bond_price = np.exp(-r * T) * 100

final_product_price = (
    price_call_ko_120 +
    price_call_ko_150 +
    price_digital_10 +
    price_digital_25 +
    price_vanilla_call
) + zero_coupon_bond_price

# Compute bank margin
bank_margin = 100 - final_product_price

stored_results = {
    "paths_digital_10": paths_digital_10,
    "paths_digital_25": paths_digital_25,
    "paths_ko_120": paths_ko_120,
    "paths_ko_150": paths_ko_150,
    "paths_vanilla_call": paths_vanilla_call,
    "final_prices_ko_120": final_prices_ko_120,
    "final_prices_ko_150": final_prices_ko_150,
    "final_prices_vanilla_call": final_prices_vanilla_call,
    "digital_10_mc": digital_10_mc,
    "digital_25_mc": digital_25_mc,
    "call_ko_120_mc": call_ko_120_mc,
    "call_ko_150_mc": call_ko_150_mc,
    "vanilla_call_mc": vanilla_call_mc
}

# =============================================================================
# Pricing Breakdown & Final Price
# =============================================================================

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<h3 style='text-align: left; color: #333;'>📊 OptiLock 40 Pricing Breakdown</h3>", unsafe_allow_html=True)

    df_pricing = pd.DataFrame({
        "Component": [
            "One-Touch Digital (120%)",
            "One-Touch Digital (150%)",
            "Knock-Out Call (100%, KO at 120%)",
            "Knock-Out Call (120%, KO at 150%)",
            "Vanilla Call (150%)",
            "Zero Coupon Bond"
        ],
        "Price (%)": [
            f"{price_digital_10:.2f}%",
            f"{price_digital_25:.2f}%",
            f"{price_call_ko_120:.2f}%",
            f"{price_call_ko_150:.2f}%",
            f"{price_vanilla_call:.2f}%",
            f"{zero_coupon_bond_price:.2f}%"
        ]
    })

    st.dataframe(df_pricing.style
        .set_properties(**{
            "text-align": "center",
            "font-size": "16px"
        })
        .set_table_styles([
            {"selector": "thead th", "props": [("background-color", "#e9ecef"), ("color", "#333"), ("font-weight", "bold"), ("text-align", "center")]},
            {"selector": "tbody td", "props": [("background-color", "#f8f9fa"), ("border", "1px solid #ddd"), ("padding", "6px")]},
            {"selector": "tbody tr:hover", "props": [("background-color", "#dee2e6")]}
        ]),
        width=700, height=260
    )

with col2:
    st.markdown("<h2 style='text-align: center; font-size: 30px;'>💰 Final Price</h2>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div style="padding: 20px; margin-top: 15px; border-radius: 12px; border: 3px solid #008000; 
                    background-color: #e6f4ea; text-align: center; font-size: 28px;">
            <strong>{final_product_price:.2f} %</strong>
        </div>
    """, unsafe_allow_html=True)
    
    bank_margin_color = "#FF4500" if bank_margin < 0 else "#008000"

    st.markdown(f"""
        <div style="padding: 20px; margin-top: 20px; border-radius: 12px; border: 3px solid {bank_margin_color}; 
                    background-color: #f8f9fa; text-align: center; font-size: 22px;">
            🏦 <strong> Bank Margin:</strong> 
            <span style="font-weight: bold; color: {bank_margin_color};">
                {bank_margin:.2f}%
            </span>
        </div>
    """, unsafe_allow_html=True)
    
st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

st.markdown("---")

st.markdown("<h3 style='text-align: center;'>🏦 Bank Margin & Buyback Price Calculation</h3>", unsafe_allow_html=True)

# Liquidity penalty selection
liquidity_penalty_percentage = st.slider(
    "Select Liquidity Penalty (%)", 
    min_value=0.0, max_value=2.0, value=1.0, step=0.1
) / 100

liquidity_penalty = liquidity_penalty_percentage * final_product_price
buyback_price = final_product_price - liquidity_penalty

bank_margin_color = "#FF4500" if bank_margin < 0 else "#008000"

st.markdown(f"""
    <style>
        .pricing-info {{
            background-color: #f8f9fa; 
            padding: 15px; 
            border-radius: 10px; 
            border: 1px solid #ddd; 
            text-align: center;
            margin: 15px;
        }}
        .pricing-value {{
            font-size: 22px; 
            font-weight: bold; 
            color: {bank_margin_color};
        }}
    </style>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div class="pricing-info">
        🏦 **Bank Upfront Margin:** <span class="pricing-value">{bank_margin:.2f}%</span><br>
        💸 **Immediate Buyback Price (with {liquidity_penalty_percentage * 100:.1f}% penalty):** 
        <span style="font-size: 22px; font-weight: bold; color: #008000;">{buyback_price:.2f}%</span>
    </div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

# =============================================================================
# Define a range of NSimul values to observe convergence
# =============================================================================

NSimul_values = np.logspace(2, np.log10(NSimul), num=10, dtype=int)

# Storage for tracking convergence
convergence_results = {
    "NSimul": [],
    "Digital_10": [],
    "Digital_25": [],
    "Call_KO_120": [],
    "Call_KO_150": [],
    "Vanilla_Call": [],
    "Full_Product_with_ZCB": []
}

# Iterate through different NSimul values
for NSimul in NSimul_values:
    
    # Retrieve stored simulation results for current NSimul
    digital_10_mc = stored_results["digital_10_mc"][:NSimul]
    digital_25_mc = stored_results["digital_25_mc"][:NSimul]
    call_ko_120_mc = stored_results["call_ko_120_mc"][:NSimul]
    call_ko_150_mc = stored_results["call_ko_150_mc"][:NSimul]
    vanilla_call_mc = stored_results["vanilla_call_mc"][:NSimul]

    # Compute expected payoffs
    expected_payoff_digital_10 = np.mean(digital_10_mc)
    expected_payoff_digital_25 = np.mean(digital_25_mc)
    expected_payoff_call_ko_120 = np.mean(call_ko_120_mc)
    expected_payoff_call_ko_150 = np.mean(call_ko_150_mc)
    expected_payoff_vanilla_call = np.mean(vanilla_call_mc)

    # Discounted Prices
    price_digital_10 = expected_payoff_digital_10 * discount_factor
    price_digital_25 = expected_payoff_digital_25 * discount_factor
    price_call_ko_120 = expected_payoff_call_ko_120 * discount_factor * Participation
    price_call_ko_150 = expected_payoff_call_ko_150 * discount_factor * Participation
    price_vanilla_call = expected_payoff_vanilla_call * discount_factor * Participation

    # Full product price
    product_price = (
        price_call_ko_120 +
        price_call_ko_150 +
        price_digital_10 +
        price_digital_25 +
        price_vanilla_call
    )

    # Store results
    convergence_results["NSimul"].append(NSimul)
    convergence_results["Digital_10"].append(price_digital_10)
    convergence_results["Digital_25"].append(price_digital_25)
    convergence_results["Call_KO_120"].append(price_call_ko_120)
    convergence_results["Call_KO_150"].append(price_call_ko_150)
    convergence_results["Vanilla_Call"].append(price_vanilla_call)
    convergence_results["Full_Product_with_ZCB"].append(product_price)

# Streamlit Visualization
st.title("Monte Carlo Convergence Analysis")
st.write("This chart shows how option prices and the structured product price converge as NSimul increases.")

convergence_df = pd.DataFrame(convergence_results)
options = list(convergence_results.keys())[1:]
selected_options = st.multiselect("Select options to display:", options, default=options)

if selected_options:
    st.line_chart(convergence_df.set_index("NSimul")[selected_options])

st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

# =============================================================================
# Black-Scholes Analytical Price Calculation
# =============================================================================
def black_scholes_call(S0, K, T, r, vol, div):
    d1 = (np.log(S0 / K) + (r - div + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return (S0 * np.exp(-div * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)) / S0 * 100

# Compute Black-Scholes price
bs_price = black_scholes_call(S0, 1.5 * S0, T, r, vol_vanilla_call, div) * Participation

discount_factor = np.exp(-r * T)

# =============================================================================
# Monte Carlo vs Black-Scholes Convergence Analysis
# =============================================================================

# Store results
bs_results = {
    "NSimul": [],
    "Vanilla_Call_MC": [],
    "Vanilla_Call_BS": [],
    "Absolute_Error": [],
    "Error_Volatility": [],
    "CI_Lower": [],
    "CI_Upper": []
}

for NSimul in NSimul_values:
    vanilla_call_mc = stored_results["vanilla_call_mc"][:NSimul]
    
    mc_price = np.mean(vanilla_call_mc) * discount_factor * Participation
    mc_std_error = np.std(vanilla_call_mc) / np.sqrt(NSimul)
    
    bs_results["NSimul"].append(NSimul)
    bs_results["Vanilla_Call_MC"].append(mc_price)
    bs_results["Vanilla_Call_BS"].append(bs_price)
    bs_results["Absolute_Error"].append(abs(mc_price - bs_price))
    bs_results["Error_Volatility"].append(mc_std_error)
    bs_results["CI_Lower"].append(mc_price - 1.96 * mc_std_error)
    bs_results["CI_Upper"].append(mc_price + 1.96 * mc_std_error)

# Convert to DataFrame
convergence_df = pd.DataFrame(bs_results)

# =============================================================================
# Streamlit Visualization
# =============================================================================

st.title("Monte Carlo vs Black-Scholes Convergence Analysis")
st.write("Comparison of Monte Carlo simulated prices against the Black-Scholes analytical solution as the number of simulations increases.")
st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])

with col1:
    # Monte Carlo Convergence to Black-Scholes
    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(x=convergence_df["NSimul"], y=convergence_df["Vanilla_Call_MC"], 
                                  mode='lines+markers', name="Monte Carlo Price", line=dict(color='blue')))
    fig_conv.add_trace(go.Scatter(x=convergence_df["NSimul"], y=convergence_df["Vanilla_Call_BS"], 
                                  mode='lines', name="Black-Scholes Price", line=dict(color='red', dash='dash')))
    fig_conv.add_trace(go.Scatter(x=convergence_df["NSimul"], y=convergence_df["CI_Lower"], 
                                  mode='lines', name="95% CI Lower", line=dict(color='gray', dash='dot')))
    fig_conv.add_trace(go.Scatter(x=convergence_df["NSimul"], y=convergence_df["CI_Upper"], 
                                  mode='lines', name="95% CI Upper", line=dict(color='gray', dash='dot')))
    
    fig_conv.update_layout(
        title="Monte Carlo Convergence to Black-Scholes Price", 
        xaxis=dict(title="Number of Simulations (log scale)", type="log"), 
        yaxis=dict(title="Option Price (%)"), 
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig_conv, use_container_width=True)

with col2:
    # Volatility of Monte Carlo Errors
    fig_error = go.Figure()
    fig_error.add_trace(go.Scatter(x=convergence_df["NSimul"], y=convergence_df["Error_Volatility"], 
                                   mode='lines+markers', name="Error Volatility", line=dict(color='orange')))
    
    fig_error.update_layout(
        title="Volatility of Monte Carlo Pricing Error", 
        xaxis=dict(title="Number of Simulations (log scale)", type="log"), 
        yaxis=dict(title="Error Volatility (%)"), 
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig_error, use_container_width=True)
st.markdown("---")

# Summary & Insights
st.markdown("""
    <h3>Summary & Insights</h3>
    <p>Our Monte Carlo simulation analysis yields the following key insights:</p>
    <ol>
        <li>The Monte Carlo price progressively converges to the Black-Scholes price as the number of simulations increases.</li>
        <li>The absolute error decreases with more simulations; however, its volatility remains significant when the number of simulations is low.</li>
        <li>A high number of simulations is required to achieve price stability with minimal error, highlighting the computational cost of Monte Carlo methods.</li>
    </ol>
""", unsafe_allow_html=True)

st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

# =============================================================================
# Sensitivity & Margin Impact Analysis
# =============================================================================

st.markdown("<h2 style='text-align: center;'> Sensitivity Analysis Dashboard</h2>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

col_left, col_middle, col_right = st.columns([3, 0.1, 2])

with col_left:
    st.markdown("<h4> Interest Rate Change</h4>", unsafe_allow_html=True)
    
    rate_change_type = st.radio(
        "Rate Change Type",
        ("Decrease", "Increase"),
        horizontal=True
    )

    basis_points_change = st.slider(
        "Interest Rate Change (in basis points)", 
        min_value=0, max_value=100, value=50, step=5
    ) / 10000

    new_selected_r = r - basis_points_change if rate_change_type == "Decrease" else r + basis_points_change
    new_selected_r = max(new_selected_r, 0.0)

    st.markdown("<hr style='border: 1px solid #ddd;'>", unsafe_allow_html=True)

    st.markdown("<h4> Volatility Adjustment</h4>", unsafe_allow_html=True)
    
    vol_change_type = st.radio(
        "Volatility Change Type",
        ("Decrease", "Increase"),
        horizontal=True
    )

    volatility_change = st.slider(
        "Volatility Change (in percentage points)", 
        min_value=0.0, max_value=5.0, value=1.0, step=0.1
    ) / 100

    st.markdown("<hr style='border: 1px solid #ddd;'>", unsafe_allow_html=True)

with col_middle:
    st.markdown(
        """<div style="border-left: 2px solid #bbb; height: 100%; margin-top: 10px;"></div>""",
        unsafe_allow_html=True
    )
    
with col_right:
    
    st.markdown("<h4> Maturity Adjustment</h4>", unsafe_allow_html=True)
                
    maturity_change = st.slider(
        "Increase in Maturity (Years)", 
        min_value=0, max_value=5, value=1, step=1
    )

    new_selected_T = T + maturity_change

    st.markdown("<hr style='border: 1px solid #ddd;'>", unsafe_allow_html=True)
    
    st.markdown("<h4> Digital One-Touch Coupon</h4>", unsafe_allow_html=True)

    digital_10_base = 10
    digital_25_base = 15

    coupon_change_type_10 = st.radio(
        "Change Type for Digital 10%", ("Decrease", "Increase"), horizontal=True, key="coupon_10"
    )

    coupon_change_10 = st.slider(
        "Change in Digital Coupon 10% (in percentage points)", 
        min_value=0.0, max_value=5.0, value=1.0, step=0.1
    )

    if coupon_change_type_10 == "Decrease":
        new_digital_10_payout = digital_10_base - coupon_change_10
    else:
        new_digital_10_payout = digital_10_base + coupon_change_10

    coupon_change_type_25 = st.radio(
        "Change Type for Digital 25%", ("Decrease", "Increase"), horizontal=True, key="coupon_25"
    )

    coupon_change_25 = st.slider(
        "Change in Digital Coupon 25% (in percentage points)", 
        min_value=0.0, max_value=5.0, value=1.0, step=0.1
    )
    
    if coupon_change_type_25 == "Decrease":
        new_digital_25_payout = digital_25_base - coupon_change_25
    else:
        new_digital_25_payout = digital_25_base + coupon_change_25



# =============================================================================
# Updated Volatilities Based on User Changes
# =============================================================================
    vol_vanilla_call_new = vol_interpolator([150, new_selected_T])[0]
    vol_digital_10_new = vol_interpolator([barrier_10_pct, new_selected_T])[0]
    vol_digital_25_new = vol_interpolator([barrier_25_pct, new_selected_T])[0]
    vol_ko_120_new = vol_interpolator([100, new_selected_T])[0]
    vol_ko_150_new = vol_interpolator([120, new_selected_T])[0]

    # Apply user-defined volatility change
    if vol_change_type == "Decrease":
        vol_vanilla_call_new -= volatility_change
        vol_digital_10_new -= volatility_change
        vol_digital_25_new -= volatility_change
        vol_ko_120_new -= volatility_change
        vol_ko_150_new -= volatility_change
    else:
        vol_vanilla_call_new += volatility_change
        vol_digital_10_new += volatility_change
        vol_digital_25_new += volatility_change
        vol_ko_120_new += volatility_change
        vol_ko_150_new += volatility_change

# =============================================================================
# Monte Carlo Simulation with Updated Parameters (Sensitivity Analysis)
# =============================================================================
np.random.seed(42)

# Update steps based on new maturity
N_steps_new = 252 * new_selected_T  
dt = new_selected_T / N_steps_new

# Initialize new paths
paths_digital_10_new = np.zeros((NSimul, N_steps_new + 1))
paths_digital_25_new = np.zeros((NSimul, N_steps_new + 1))
paths_ko_120_new = np.zeros((NSimul, N_steps_new + 1))
paths_ko_150_new = np.zeros((NSimul, N_steps_new + 1))
paths_vanilla_call_new = np.zeros((NSimul, N_steps_new + 1))

# Set initial prices
paths_digital_10_new[:, 0] = S0
paths_digital_25_new[:, 0] = S0
paths_ko_120_new[:, 0] = S0
paths_ko_150_new[:, 0] = S0
paths_vanilla_call_new[:, 0] = S0

# Generate new random shocks for updated simulation
Z = np.random.standard_normal((NSimul, N_steps_new))

# **Use your function to simulate updated paths**
simulate_paths(paths_digital_10_new, new_selected_r, vol_digital_10_new, div)
simulate_paths(paths_digital_25_new, new_selected_r, vol_digital_25_new, div)
simulate_paths(paths_ko_120_new, new_selected_r, vol_ko_120_new, div)
simulate_paths(paths_ko_150_new, new_selected_r, vol_ko_150_new, div)
simulate_paths(paths_vanilla_call_new, new_selected_r, vol_vanilla_call_new, div)

# Compute max path to check barriers (One-Touch Condition)
max_paths_digital_10_new = np.max(paths_digital_10_new, axis=1)
max_paths_digital_25_new = np.max(paths_digital_25_new, axis=1)

hit_10_new = (max_paths_digital_10_new >= Barrier_10).astype(int)  
hit_25_new = (max_paths_digital_25_new >= Barrier_25).astype(int)

# =============================================================================
# Updated Payoff Computation for Each Option
# =============================================================================
final_prices_ko_120_new = paths_ko_120_new[:, -1]
final_prices_ko_150_new = paths_ko_150_new[:, -1]
final_prices_vanilla_call_new = paths_vanilla_call_new[:, -1]

# **Use adjusted digital one-touch payouts**
digital_10_mc_new = np.where(hit_10_new == 1, new_digital_10_payout, 0)  
digital_25_mc_new = np.where(hit_25_new == 1, new_digital_25_payout, 0)

# Knock-Out Call Options
call_ko_120_mc_new = np.where(hit_10_new == 0,
                               np.maximum(Participation * (final_prices_ko_120_new - S0) / S0, 0) * 100, 0)

call_ko_150_mc_new = np.where((hit_10_new == 1) & (hit_25_new == 0),
                               np.maximum(Participation * (final_prices_ko_150_new - 1.2 * S0) / S0, 0) * 100, 0)

# Vanilla Call
vanilla_call_mc_new = np.maximum((final_prices_vanilla_call_new - 1.5 * S0) / S0, 0) * 100

# =============================================================================
# Compute Updated Expected Payoff & Discount Factor
# =============================================================================
expected_payoff_digital_10_new = np.mean(digital_10_mc_new)
expected_payoff_digital_25_new = np.mean(digital_25_mc_new)
expected_payoff_call_ko_120_new = np.mean(call_ko_120_mc_new)
expected_payoff_call_ko_150_new = np.mean(call_ko_150_mc_new)
expected_payoff_vanilla_call_new = np.mean(vanilla_call_mc_new)

# Compute Discount Factor
discount_factor_new = np.exp(-new_selected_r * new_selected_T)

# Discounted Option Prices
price_digital_10_new = expected_payoff_digital_10_new * discount_factor_new
price_digital_25_new = expected_payoff_digital_25_new * discount_factor_new
price_call_ko_120_new = expected_payoff_call_ko_120_new * discount_factor_new * Participation
price_call_ko_150_new = expected_payoff_call_ko_150_new * discount_factor_new * Participation
price_vanilla_call_new = expected_payoff_vanilla_call_new * discount_factor_new * Participation

# =============================================================================
# Final Structured Product Pricing (Updated)
# =============================================================================
final_product_price_new = (
    price_call_ko_120_new +
    price_call_ko_150_new +
    price_digital_10_new +
    price_digital_25_new +
    price_vanilla_call_new
) + np.exp(-new_selected_r * new_selected_T) * 100

# Compute Bank Margin
bank_margin_new = 100 - final_product_price_new

# =============================================================================
# Display Updated Results
# =============================================================================
st.markdown("---")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<h3 style='text-align: left; color: #333;'>📊 Updated OptiLock 40 Pricing Breakdown</h3>", unsafe_allow_html=True)
    
    df_updated_pricing = pd.DataFrame({
        "Component": [
            "One-Touch Digital (120%)",
            "One-Touch Digital (150%)",
            "Knock-Out Call (100%, KO at 120%)",
            "Knock-Out Call (120%, KO at 150%)",
            "Vanilla Call (150%)",
            "Zero Coupon Bond"
        ],
        "Updated Price (%)": [
            f"{price_digital_10_new:.2f}%",
            f"{price_digital_25_new:.2f}%",
            f"{price_call_ko_120_new:.2f}%",
            f"{price_call_ko_150_new:.2f}%",
            f"{price_vanilla_call_new:.2f}%",
            f"{np.exp(-new_selected_r * new_selected_T) * 100:.2f}%"
        ]
    })
    
    st.dataframe(df_updated_pricing.style.set_properties(**{
        "text-align": "center",
        "background-color": "#f8f9fa",
        "border": "1px solid #ddd",
        "font-size": "16px"
    }), width=600)

with col2:
    st.markdown("<h2 style='text-align: center; font-size: 30px;'>💰 Updated Final Price</h2>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div style="padding: 20px; margin-top: 15px; border-radius: 12px; border: 3px solid #008000; 
                    background-color: #e6f4ea; text-align: center; font-size: 28px;">
            <strong>{final_product_price_new:.2f} %</strong>
        </div>
    """, unsafe_allow_html=True)
    
    bank_margin_color = "#FF4500" if bank_margin_new < 0 else "#008000"

    st.markdown(f"""
        <div style="padding: 20px; margin-top: 20px; border-radius: 12px; border: 3px solid {bank_margin_color}; 
                    background-color: #f8f9fa; text-align: center; font-size: 22px;">
            🏦 <strong>Updated Bank Margin:</strong> 
            <span style="font-weight: bold; color: {bank_margin_color};">
                {bank_margin_new:.2f}%
            </span>
        </div>
    """, unsafe_allow_html=True)
    
st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

# =============================================================================
# Key Takeaways & Strategic Insights
# =============================================================================

st.markdown("<h3 style='text-align: center; color: #222;'>Key Takeaways & Strategic Insights</h3>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

text_style = "font-size:16px; line-height:1.5; color:#222;"
section_style = "background-color:#f1f3f5; padding:14px; border-radius:6px; margin-bottom:10px;"

st.markdown(f"""
    <div style='{section_style}'>
        <h4 style='color:#333; margin-bottom:5px;'>Interest Rate Sensitivity</h4>
        <p style='{text_style}'>A decline in interest rates reduces the Zero Coupon Bond (ZCB) value, impacting overall margin.</p>
        <p style='{text_style}'><b>Mitigation:</b> Interest rate swaps and duration hedging can help manage risk exposure.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div style='{section_style}'>
        <h4 style='color:#333; margin-bottom:5px;'>Maturity Extension</h4>
        <p style='{text_style}'>Extending duration enhances long-term stability but increases reinvestment and interest rate risks.</p>
        <p style='{text_style}'><b>Strategy:</b> Adjusting ZCB allocation and optimizing asset-liability duration can balance the risk-return tradeoff.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div style='{section_style}'>
        <h4 style='color:#333; margin-bottom:5px;'>Coupon Adjustments</h4>
        <p style='{text_style}'>Higher coupons enhance product attractiveness but reduce bank margins.</p>
        <p style='{text_style}'><b>Optimization:</b> Structured coupons (e.g., step-up, floating rates) offer flexibility while managing risk exposure.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div style='{section_style}'>
        <h4 style='color:#333; margin-bottom:5px;'>Volatility Impact</h4>
        <p style='{text_style}'>Increased market volatility raises option prices, affecting structured product profitability.</p>
        <p style='{text_style}'><b>Risk Control:</b> Implementing dynamic hedging techniques helps smooth pricing fluctuations.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div style='{section_style}'>
        <h4 style='color:#333; margin-bottom:5px;'>Scenario Planning & Stress Testing</h4>
        <p style='{text_style}'>Simulating multiple rate, volatility, and market conditions ensures robustness against adverse movements.</p>
        <p style='{text_style}'><b>Best Practice:</b> Regular Monte Carlo simulations and historical backtesting improve risk-adjusted decision-making.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

# =============================================================================
# Black-Scholes Analytical Price Calculation (After Sensitivity Adjustments)
# =============================================================================
def black_scholes_call_2(S0, K, T, r, vol, div):
    d1 = (np.log(S0 / K) + (r - div + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return (S0 * np.exp(-div * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)) / S0 * 100

# Compute updated Black-Scholes price
bs_price_new = black_scholes_call_2(S0, 1.5 * S0, new_selected_T, new_selected_r, vol_vanilla_call_new, div) * Participation

# Compute updated discount factor
discount_factor_new = np.exp(-new_selected_r * new_selected_T)

# =============================================================================
# Monte Carlo vs Black-Scholes Convergence Analysis (Updated)
# =============================================================================

# Store updated results
bs_results_new = {
    "NSimul": [],
    "Vanilla_Call_MC": [],
    "Vanilla_Call_BS": [],
    "Absolute_Error": [],
    "Error_Volatility": [],
    "CI_Lower": [],
    "CI_Upper": []
}

for NSimul in NSimul_values:
    vanilla_call_mc_new_subset = vanilla_call_mc_new[:NSimul]
    
    mc_price_new = np.mean(vanilla_call_mc_new_subset) * discount_factor_new * Participation
    
    mc_std_error_new = np.std(vanilla_call_mc_new_subset) / np.sqrt(NSimul)

    bs_results_new["NSimul"].append(NSimul)
    bs_results_new["Vanilla_Call_MC"].append(mc_price_new)
    bs_results_new["Vanilla_Call_BS"].append(bs_price_new)
    bs_results_new["Absolute_Error"].append(abs(mc_price_new - bs_price_new))
    bs_results_new["Error_Volatility"].append(mc_std_error_new)
    bs_results_new["CI_Lower"].append(mc_price_new - 1.96 * mc_std_error_new)
    bs_results_new["CI_Upper"].append(mc_price_new + 1.96 * mc_std_error_new)

convergence_df_new = pd.DataFrame(bs_results_new)

# =============================================================================
# Streamlit Visualization (Updated)
# =============================================================================
st.title("Monte Carlo vs Black-Scholes Convergence Analysis (Updated)")
st.write("Comparison of Monte Carlo simulated prices against the Black-Scholes analytical solution after sensitivity adjustments.")
st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])

with col1:
    fig_conv_new = go.Figure()
    fig_conv_new.add_trace(go.Scatter(x=convergence_df_new["NSimul"], y=convergence_df_new["Vanilla_Call_MC"], 
                                      mode='lines+markers', name="Monte Carlo Price", line=dict(color='blue')))
    fig_conv_new.add_trace(go.Scatter(x=convergence_df_new["NSimul"], y=convergence_df_new["Vanilla_Call_BS"], 
                                      mode='lines', name="Black-Scholes Price", line=dict(color='red', dash='dash')))
    fig_conv_new.add_trace(go.Scatter(x=convergence_df_new["NSimul"], y=convergence_df_new["CI_Lower"], 
                                      mode='lines', name="95% CI Lower", line=dict(color='gray', dash='dot')))
    fig_conv_new.add_trace(go.Scatter(x=convergence_df_new["NSimul"], y=convergence_df_new["CI_Upper"], 
                                      mode='lines', name="95% CI Upper", line=dict(color='gray', dash='dot')))
    
    fig_conv_new.update_layout(
        title="Monte Carlo Convergence to Black-Scholes Price (After Sensitivity)", 
        xaxis=dict(title="Number of Simulations (log scale)", type="log"), 
        yaxis=dict(title="Option Price (%)"), 
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig_conv_new, use_container_width=True)

with col2:
    fig_error_new = go.Figure()
    fig_error_new.add_trace(go.Scatter(x=convergence_df_new["NSimul"], y=convergence_df_new["Error_Volatility"], 
                                       mode='lines+markers', name="Error Volatility", line=dict(color='orange')))
    
    fig_error_new.update_layout(
        title="Volatility of Monte Carlo Pricing Error (After Sensitivity)", 
        xaxis=dict(title="Number of Simulations (log scale)", type="log"), 
        yaxis=dict(title="Error Volatility (%)"), 
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig_error_new, use_container_width=True)

st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)
# =============================================================================
# Monte Carlo Simulation of CAC 40 Prices
# =============================================================================
st.markdown("<h2 style='text-align: center;'>Monte Carlo Simulation - Analysis</h2>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

# =============================================================================
# Final CAC 40 Price Distribution
# =============================================================================

prob_hit_120_new = np.mean(hit_10_new) * 100
prob_hit_150_new = np.mean(hit_25_new) * 100

col1, col2 = st.columns([2.5, 1.5])

with col1:
    st.markdown("")

    density_new = gaussian_kde(final_prices_vanilla_call_new)
    x_vals_new = np.linspace(min(final_prices_vanilla_call_new), max(final_prices_vanilla_call_new), 200)
    density_vals_new = density_new(x_vals_new)

    fig_dist = go.Figure()

    fig_dist.add_trace(go.Histogram(
        x=final_prices_vanilla_call_new,
        nbinsx=50,
        histnorm='probability density',
        marker=dict(color="rgba(0, 0, 255, 0.5)", line=dict(color="black", width=0.3)),
        opacity=0.8,
        name="Simulated Final CAC 40 Prices"
    ))

    fig_dist.add_trace(go.Scatter(
        x=x_vals_new,
        y=density_vals_new,
        mode="lines",
        line=dict(color="red", width=2),
        name="Density Estimation"
    ))

    fig_dist.add_trace(go.Scatter(
        x=[S0, S0], y=[0, max(density_vals_new) * 1.1],
        mode="lines", name="Initial CAC 40 Level",
        line=dict(color="red", dash="dash")
    ))

    fig_dist.add_trace(go.Scatter(
        x=[Barrier_10, Barrier_10], y=[0, max(density_vals_new) * 1.1],
        mode="lines", name=f"120% Barrier",
        line=dict(color="green", dash="dash")
    ))

    fig_dist.add_trace(go.Scatter(
        x=[Barrier_25, Barrier_25], y=[0, max(density_vals_new) * 1.1],
        mode="lines", name=f"150% Barrier",
        line=dict(color="purple", dash="dash")
    ))

    fig_dist.update_layout(
        xaxis_title="Final CAC 40 Level at Maturity",
        yaxis_title="Density",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top", y=1,
            xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
            font=dict(size=14)
        ),
        height=400,
        margin=dict(t=20, b=20, l=50, r=50)
    )

    st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    st.markdown("<h4 style='text-align: left;'>📊 Barrier Activation Probabilities</h4>", unsafe_allow_html=True)

    barrier_stats_df_new = pd.DataFrame({
        "Barrier Level": ["120%", "150%"],
        "Probability (%)": [f"{prob_hit_120_new:.2f}%", f"{prob_hit_150_new:.2f}%"]
    })

    st.dataframe(barrier_stats_df_new.style.set_properties(**{"text-align": "center"}), 
                 hide_index=True, width=350)

    st.markdown("<h4 style='text-align: left; margin-top: 10px;'>📊 Key Statistics</h4>", unsafe_allow_html=True)

    key_stats_df = pd.DataFrame({
        "Metric": ["Min Price", "Max Price", "Mean Price"],
        "Value": [f"{final_prices_vanilla_call_new.min():,.2f}", 
                  f"{final_prices_vanilla_call_new.max():,.2f}", 
                  f"{final_prices_vanilla_call_new.mean():,.2f}"]
    })

    st.dataframe(key_stats_df.style.set_properties(**{"text-align": "center"}), 
                 hide_index=True, width=350)

st.markdown("---")

# =============================================================================
# Payoff Summary at Maturity
# =============================================================================
st.markdown("<h3 style='text-align: center;'>Payoff Summary at Maturity</h3>", unsafe_allow_html=True)

payoff_df_new = pd.DataFrame({
    "Final CAC 40 Level": final_prices_vanilla_call_new,
    "One-Touch 120% Payoff": digital_10_mc_new,
    "One-Touch 150% Payoff": digital_25_mc_new,
    "Knock-Out Call (100%, KO at 120%)": call_ko_120_mc_new,
    "Knock-Out Call (100%, KO at 150%)": call_ko_150_mc_new,
    "Vanilla Call (150%)": vanilla_call_mc_new,
    "Total Payoff": (
        digital_10_mc_new +
        digital_25_mc_new +
        call_ko_120_mc_new +
        call_ko_150_mc_new +
        vanilla_call_mc_new
    )
}).round(2)

MAX_ROWS_DISPLAY = 5000
payoff_sample_df_new = payoff_df_new.head(MAX_ROWS_DISPLAY)

payoff_sample_df_new = payoff_sample_df_new.map(lambda x: f"{x:.2f}")

styled_payoff_df_new = payoff_sample_df_new.style.map(
    lambda x: 'background-color: #ffcccb' if float(x) > 0 else '', subset=["Total Payoff"]
)

st.dataframe(styled_payoff_df_new, width=1000, height=200)
st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

# =============================================================================
# Payoff Visualization with Backtesting Spot Selection
# =============================================================================
st.markdown("<h3 style='text-align: center;'>Payoff Visualization</h3>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 2], gap="large")

with col1:
    st.markdown("<h4 style='margin-bottom: 10px;'>CAC 40 Variation (%)</h4>", unsafe_allow_html=True)
    
    cac_level = st.slider(
        "Adjust CAC 40 Variation",  
        min_value=-50, max_value=150, value=0, step=1,
        format="%d%%"
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

with col2:
    st.markdown("<h4 style='margin-top: 10px;'>Barrier Activation</h4>", unsafe_allow_html=True)

    c1, c2 = st.columns([0.2, 1])
    with c1:
        knock_120_flag = st.checkbox("", key="knock_120")
    with c2:
        st.write("✅ **120% Barrier Reached**" if knock_120_flag else "120% Barrier Not Reached")

    c3, c4 = st.columns([0.2, 1])
    with c3:
        knock_150_flag = st.checkbox("", key="knock_150")
    with c4:
        st.write("✅ **150% Barrier Reached**" if knock_150_flag else "150% Barrier Not Reached")

st.markdown("---")

# =============================================================================
# Dynamic Payoff Calculation
# =============================================================================
def calculate_payoff(cac_variation, knock_120, knock_150):
    payoff_participation = max(Participation * cac_variation, 0)
    payoff_digital_120 = 10 if knock_120 else 0
    payoff_digital_150 = 25 if knock_150 else 0
    return max(payoff_participation, payoff_digital_120, payoff_digital_150)

# Compute payoffs for the selected CAC 40 level
payoff_client = calculate_payoff(cac_level, knock_120_flag, knock_150_flag)
payoff_bank = -payoff_client

# Generate payoff evolution over different CAC 40 variations
cac_variation = np.linspace(-50, 150, 500)
overall_payoff_client = np.array([calculate_payoff(cac, knock_120_flag, knock_150_flag) for cac in cac_variation])
overall_payoff_bank = -overall_payoff_client

# =============================================================================
# Payoff Visualization
# =============================================================================

# --- First Graph: Payoff vs CAC 40 variation ---
fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=cac_variation, y=overall_payoff_client, 
    mode="lines", name="💰 Client Payoff",
    line=dict(color="green", width=2)
))

fig1.add_trace(go.Scatter(
    x=cac_variation, y=overall_payoff_bank, 
    mode="lines", name="🏦 Bank Payoff",
    line=dict(color="red", width=2)
))

fig1.add_hline(y=0, line=dict(color="black", width=1, dash="dash"))
fig1.add_vline(x=0, line=dict(color="black", width=1, dash="dash"))

fig1.update_layout(
    title="Payoff Evolution vs CAC 40 Variation",
    xaxis_title="CAC 40 Variation (%)",
    yaxis_title="Payoff (% of Initial Capital)",
    template="plotly_dark",
    hovermode="x unified",
    showlegend=True
)

st.plotly_chart(fig1)

st.markdown("---")

# --- Second Graph: Payoff for Client & Bank ---
fig2 = go.Figure()

fig2.add_trace(go.Bar(
    x=["Client", "Bank"], y=[payoff_client, payoff_bank],
    marker=dict(color=["green", "red"]),
    text=[f"{payoff_client:.2f}%", f"{payoff_bank:.2f}%"],
    textposition="inside",
    insidetextanchor="middle",
    name="Payoff Distribution"
))

fig2.add_hline(y=0, line=dict(color="black", width=1, dash="dash"))

fig2.update_layout(
    title=f"OptiLock 40 Payoff (CAC 40: {cac_level:.1f}%)",
    xaxis_title="Counterparties",
    yaxis_title="Payoff (% of Initial Capital)",
    template="plotly_dark",
    hovermode="x unified",
    showlegend=False,
    margin=dict(t=40, b=40, l=50, r=50)
)

st.plotly_chart(fig2)
st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

# =============================================================================
# Historical CAC 40 Price & Payoff Visualization
# =============================================================================
    
st.markdown("<h3 style='text-align: center;'>📊 Historical CAC 40 & Payoff Backtesting</h3>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

spot_level = st.number_input(
    "📍 Select Spot Level for Backtesting", 
    min_value=0, 
    max_value=20000,
    value=S0,
    step=50
)
    
# Download CAC 40 Data
cac40 = yf.download("^FCHI", period="5y", interval="1d")["Close"]

if cac40.empty:
    st.error("Error: No data retrieved for CAC 40. Please check the ticker symbol or internet connection.")
else:
    
    user_S0 = spot_level
    user_Barrier_10 = (barrier_10_pct / 100) * user_S0
    user_Barrier_25 = (barrier_25_pct / 100) * user_S0
    user_Participation = Participation

    # =============================================================================
    # Time-Dependent Payoff Calculation for Client
    # =============================================================================
    
    # Compute CAC 40 return based on user-defined initial level
    cac_return = (cac40 - user_S0) / user_S0 * 100

    # Track when each barrier was first hit
    knock_120_date = cac40[cac40 >= user_Barrier_10].index.min()
    knock_150_date = cac40[cac40 >= user_Barrier_25].index.min()

    # Initialize payoff series with participation-based return
    payoff_client = np.maximum(user_Participation * cac_return, 0)

    # Create boolean arrays tracking if barriers have been hit up to each date
    knock_120_active = (cac40 >= user_Barrier_10).cummax()
    knock_150_active = (cac40 >= user_Barrier_25).cummax()

    # Apply digital payoffs **only if** the barrier has been hit at that time
    payoff_client[knock_120_active] = np.maximum(payoff_client[knock_120_active], 10)
    payoff_client[knock_150_active] = np.maximum(payoff_client[knock_150_active], 25)

    # Compute P/L for Bank
    payoff_bank = -payoff_client
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(cac40.index, cac40, label="CAC 40 Price", color="blue", linewidth=1.5, alpha=0.9)
    
    ax1.axhspan(user_Barrier_10 * 0.98, user_Barrier_10 * 1.02, color="green", alpha=0.1)
    ax1.axhspan(user_Barrier_25 * 0.98, user_Barrier_25 * 1.02, color="purple", alpha=0.1)
    
    ax1.axhline(user_S0, color="black", linestyle="--", linewidth=1, label="Initial CAC 40 Level")
    ax1.axhline(user_Barrier_10, color="green", linestyle="--", linewidth=1, label=f"{barrier_10_pct}% Barrier")
    ax1.axhline(user_Barrier_25, color="purple", linestyle="--", linewidth=1, label=f"{barrier_25_pct}% Barrier")
    
    ax1.set_xlabel("Date", fontsize=13, fontweight="bold")
    ax1.set_ylabel("CAC 40 Price", fontsize=13, fontweight="bold")
    
    ax1.legend(loc="upper left", fontsize=11, frameon=True, fancybox=True, framealpha=1, edgecolor="black", facecolor="white")
    
    ax1.grid(alpha=0.4, linestyle="dotted")
    
    plt.xticks(rotation=20, fontsize=11)
    plt.yticks(fontsize=11)
    
    st.pyplot(fig)
    st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

    # =============================================================================
    # Display P/L for Client and Bank
    # =============================================================================
    st.markdown("<h3 style='text-align: center;'>💰 P/L Calculation for Client and Bank</h3>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])

    
    with col1:
        selected_date = st.date_input(
            "📅 Select a Date",
            value=cac40.index[-1],
            min_value=cac40.index[0],
            max_value=cac40.index[-1],
            format="YYYY-MM-DD"
        )
                
        selected_date = pd.to_datetime(selected_date)
        if selected_date not in cac40.index:
            selected_date = cac40.index.asof(selected_date)
        
        # Extract P/L values for the selected date
        selected_cac_return = cac_return.loc[selected_date].item()
        selected_payoff_client = payoff_client.loc[selected_date].item()
        selected_payoff_bank = payoff_bank.loc[selected_date].item()
        
        pnl_results = pd.DataFrame({
            "Metric": ["📊 CAC 40 Return (%)", "💰 Client Payoff (%)", "🏦 Bank Payoff (%)"],
            "Value": [f"{selected_cac_return:.2f}%", f"{selected_payoff_client:.2f}%", f"{selected_payoff_bank:.2f}%"]
        })
        
        st.markdown("<h4 style='text-align: center; margin-bottom: 10px;'>📊 P/L Summary</h4>", unsafe_allow_html=True)
        
        st.dataframe(
            pnl_results.style.set_properties(**{
                "text-align": "center",
                "background-color": "#ffffff",
                "border": "1px solid #ddd",
                "font-size": "14px",
                "border-radius": "8px"
            }),
            hide_index=True, width=400
        )

    with col2:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        ax2.plot(cac40.index, payoff_client, label="Client Payoff (%)", color="green", linewidth=1.5)
        ax2.plot(cac40.index, payoff_bank, label="Bank Payoff (%)", color="red", linewidth=1.5)
        
        ax2.axvline(selected_date, color="black", linestyle="--", linewidth=1, alpha=0.7, label="Selected Date")
        
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
        
        ax2.set_xlabel("Date", fontsize=13, fontweight="bold")
        ax2.set_ylabel("Payoff (% of Initial Capital)", fontsize=13, fontweight="bold")
        ax2.set_title("Client vs. Bank P/L Over Time", fontsize=15, fontweight="bold")
        
        ax2.legend(loc="upper left", fontsize=11, frameon=True, fancybox=True, edgecolor="black", facecolor="white")
        
        ax2.grid(alpha=0.4, linestyle="dotted")
        
        plt.xticks(rotation=20, fontsize=11)
        plt.yticks(fontsize=11)
        
        st.pyplot(fig2)
    
    st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

# =============================================================================
# Secondary Market Pricing
# =============================================================================

st.markdown("<h2 style='text-align: center;'>📈 Secondary Market Pricing</h2>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

st.markdown("### Market Parameters\n _Select the risk-free rates below:_")

col1, col2 = st.columns([2, 2])

with col1:
    ZC1 = st.number_input(
        "Short-Term Risk-Free Rate (%)", 
        value=3.75, min_value=0.00, max_value=10.00, step=0.01
    ) / 100
    
with col2:
    ZC3 = st.number_input(
        "Long-Term Risk-Free Rate (%)", 
        value=4.55, min_value=0.00, max_value=10.00, step=0.01
    ) / 100

st.markdown("---")

F_T1 = S0 * np.exp((ZC1 - div) * 1)
F_1y3y= (ZC3 * T - ZC1 * 1)/(T-1)
new_T_2 = T - 1

vol_vanilla_call_new_2 = vol_interpolator([150, new_T_2])[0]  # Volatility at 150% strike for 2Y
vol_digital_10_new_2 = vol_interpolator([barrier_10_pct, new_T_2])[0]  # Volatility at 120% strike for 2Y
vol_digital_25_new_2 = vol_interpolator([barrier_25_pct, new_T_2])[0]  # Volatility at 150% strike for 2Y
vol_ko_120_new_2 = vol_interpolator([100, new_T_2])[0]  # Volatility at 100% strike for 2Y
vol_ko_150_new_2 = vol_interpolator([120, new_T_2])[0]  # Volatility at 120% strike for 2Y

# =============================================================================
# Monte Carlo Simulation for Remaining Maturity (T-1 Years)
# =============================================================================

N_steps_new_2 = 252 * new_T_2
dt_new = new_T_2 / N_steps_new_2

paths_digital_10_new_2 = np.zeros((NSimul, N_steps_new + 1))
paths_digital_25_new_2 = np.zeros((NSimul, N_steps_new + 1))
paths_ko_120_new_2 = np.zeros((NSimul, N_steps_new + 1))
paths_ko_150_new_2 = np.zeros((NSimul, N_steps_new + 1))
paths_vanilla_call_new_2 = np.zeros((NSimul, N_steps_new + 1))

paths_digital_10_new_2[:, 0] = F_T1
paths_digital_25_new_2[:, 0] = F_T1
paths_ko_120_new_2[:, 0] = F_T1
paths_ko_150_new_2[:, 0] = F_T1
paths_vanilla_call_new_2[:, 0] = F_T1

Z_new = np.random.standard_normal((NSimul, N_steps_new))

simulate_paths(paths_digital_10_new_2, F_1y3y, vol_digital_10_new_2, div)
simulate_paths(paths_digital_25_new_2, F_1y3y, vol_digital_25_new_2, div)
simulate_paths(paths_ko_120_new_2, F_1y3y, vol_ko_120_new_2, div)
simulate_paths(paths_ko_150_new_2, F_1y3y, vol_ko_150_new_2, div)
simulate_paths(paths_vanilla_call_new_2, F_1y3y, vol_vanilla_call_new_2, div)

final_prices_digital_10_new_2 = paths_digital_10_new_2[:, -1]
final_prices_digital_25_new_2 = paths_digital_25_new_2[:, -1]
final_prices_ko_120_new_2 = paths_ko_120_new_2[:, -1]
final_prices_ko_150_new_2 = paths_ko_150_new_2[:, -1]
final_prices_vanilla_call_new_2 = paths_vanilla_call_new_2[:, -1]

hit_10_new_2 = (final_prices_digital_10_new_2 >= Barrier_10).astype(int)
hit_25_new_2 = (final_prices_digital_25_new_2 >= Barrier_25).astype(int)

# =============================================================================
# Recalculate Expected Payoffs
# =============================================================================

digital_10_mc_new_2 = np.where(hit_10_new_2 == 1, 10, 0)
digital_25_mc_new_2 = np.where(hit_25_new_2 == 1, 15, 0)

call_ko_120_mc_new_2 = np.where(hit_10_new_2 == 0,
                               np.maximum(Participation * (final_prices_ko_120_new_2 - S0) / S0, 0) * 100, 0)

call_ko_150_mc_new_2 = np.where((hit_10_new_2 == 1) & (hit_25_new_2 == 0),
                               np.maximum(Participation * (final_prices_ko_150_new_2 - 1.2 * S0) / S0, 0) * 100, 0)

vanilla_call_mc_new_2 = np.maximum((final_prices_vanilla_call_new_2 - 1.5 * S0) / S0, 0) * 100

expected_payoff_digital_10_new_2 = np.mean(digital_10_mc_new_2)
expected_payoff_digital_25_new_2 = np.mean(digital_25_mc_new_2)
expected_payoff_call_ko_120_new_2 = np.mean(call_ko_120_mc_new_2)
expected_payoff_call_ko_150_new_2 = np.mean(call_ko_150_mc_new_2)
expected_payoff_vanilla_call_new_2 = np.mean(vanilla_call_mc_new_2)

discount_factor_options = np.exp(-F_1y3y * (T-2))
discount_factor_zcb = np.exp(-F_1y3y * (T-1))
zero_coupon_bond_price_new_2 = discount_factor_zcb * 100

price_digital_10_new_2 = expected_payoff_digital_10_new_2 * discount_factor_options
price_digital_25_new_2 = expected_payoff_digital_25_new_2 * discount_factor_options
price_call_ko_120_new_2 = expected_payoff_call_ko_120_new_2 * discount_factor_options * Participation
price_call_ko_150_new_2 = expected_payoff_call_ko_150_new_2 * discount_factor_options * Participation
price_vanilla_call_new_2 = expected_payoff_vanilla_call_new_2 * discount_factor_options * Participation

final_product_price_new_2 = (
    price_call_ko_120_new_2 +
    price_call_ko_150_new_2 +
    price_digital_10_new_2 +
    price_digital_25_new_2 +
    price_vanilla_call_new_2
) + zero_coupon_bond_price_new_2

col1, col2 = st.columns([2, 1])

new_bank_margin=100-final_product_price_new_2

with col1:
    st.markdown("<div style='font-size:20px; font-weight:bold;'>📊 OptiLock 40 Pricing Breakdown & Evolution</div>", unsafe_allow_html=True)

    df_pricing = pd.DataFrame({
        "Component": [
            "One-Touch Digital (120%)",
            "One-Touch Digital (150%)",
            "Knock-Out Call (100%, KO at 120%)",
            "Knock-Out Call (120%, KO at 150%)",
            "Vanilla Call (150%)",
            "Zero Coupon Bond"
        ],
        "Initial Price (%)": [
            price_digital_10,
            price_digital_25,
            price_call_ko_120,
            price_call_ko_150,
            price_vanilla_call,
            zero_coupon_bond_price
        ],
        "Price in 1 year (%)": [
            price_digital_10_new_2,
            price_digital_25_new_2,
            price_call_ko_120_new_2,
            price_call_ko_150_new_2,
            price_vanilla_call_new_2,
            zero_coupon_bond_price_new_2
        ]
    })

    df_pricing["Initial Price (%)"] = pd.to_numeric(df_pricing["Initial Price (%)"], errors="coerce")
    df_pricing["Price in 1 year (%)"] = pd.to_numeric(df_pricing["Price in 1 year (%)"], errors="coerce")

    df_pricing["Change (%)"] = df_pricing["Price in 1 year (%)"] - df_pricing["Initial Price (%)"]

    df_pricing["Initial Price (%)"] = df_pricing["Initial Price (%)"].map(lambda x: f"{x:.2f}%")
    df_pricing["Price in 1 year (%)"] = df_pricing["Price in 1 year (%)"].map(lambda x: f"{x:.2f}%")
    df_pricing["Change (%)"] = df_pricing["Change (%)"].map(lambda x: f"{x:+.2f}%")

    def highlight_changes(val):
        try:
            val_num = float(val.replace("%", ""))
            color = "green" if val_num > 0 else "red"
            return f"color: {color}; font-weight: bold;"
        except:
            return ""

    st.dataframe(df_pricing.style.map(highlight_changes, subset=["Change (%)"]), use_container_width=True)

with col2:
    st.markdown("<div style='font-size:20px; font-weight:bold;'>💰 Final Product Price in 1 year</div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="padding: 20px; margin-top: 15px; border-radius: 12px; border: 3px solid #008000; 
                    background-color: #e6f4ea; text-align: center; font-size: 28px;">
            <strong>{final_product_price_new_2:.2f} %</strong>
        </div>
    """, unsafe_allow_html=True)
    
    # Bank Margin Section
    bank_margin_color = "#FF4500" if bank_margin < 0 else "#008000"

    st.markdown(f"""
        <div style="padding: 20px; margin-top: 20px; border-radius: 12px; border: 3px solid {bank_margin_color}; 
                    background-color: #f8f9fa; text-align: center; font-size: 22px;">
            🏦 <strong> New Bank Margin:</strong> 
            <span style="font-weight: bold; color: {bank_margin_color};">
                {new_bank_margin:.2f}%
            </span>
        </div>
    """, unsafe_allow_html=True)
   
# =============================================================================
# Bid-Ask Spread Impact & Buyback Price
# =============================================================================

st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)
st.markdown("### 🏦 Bank Margin & Buyback Price")

rate_shift = st.number_input("Rate Shift (bps)", value=4, step=1)
vol_shift = st.number_input("Volatility Shift (%)", value=2.0, step=0.1)

# Adjusted interest rates (shifting by ±4bp)
ZC1_up = ZC1 + (rate_shift/10000)
ZC1_down = ZC1 - (rate_shift/10000)
ZC3_up = ZC3 + (rate_shift/10000)
ZC3_down = ZC3 - (rate_shift/10000)

F_T1_up = S0 * np.exp((ZC1_up - div) * 1)
F_T1_down = S0 * np.exp((ZC1_down - div) * 1)

# Compute forward rates with new interest rates
F_1y3y_up = (ZC3_up * T - ZC1_up * 1) / (T - 1)
F_1y3y_down = (ZC3_down * T - ZC1_down * 1) / (T - 1)

# Adjust volatilities (shifting by ±2%)
vol_factor_up = (1+vol_shift/100)
vol_factor_down = (1-vol_shift/100)

vol_vanilla_call_up = vol_vanilla_call_new_2 * vol_factor_up
vol_vanilla_call_down = vol_vanilla_call_new_2 * vol_factor_down

vol_digital_10_up = vol_digital_10_new_2 * vol_factor_up
vol_digital_10_down = vol_digital_10_new_2 * vol_factor_down

vol_digital_25_up = vol_digital_25_new_2 * vol_factor_up
vol_digital_25_down = vol_digital_25_new_2 * vol_factor_down

vol_ko_120_up = vol_ko_120_new_2 * vol_factor_up
vol_ko_120_down = vol_ko_120_new_2 * vol_factor_down

vol_ko_150_up = vol_ko_150_new_2 * vol_factor_up
vol_ko_150_down = vol_ko_150_new_2 * vol_factor_down

N_steps_new_2 = 252 * new_T_2
dt_new = new_T_2 / N_steps_new_2

# Initialize paths for shifted rates and volatilities
paths_digital_10_up = np.zeros((NSimul, N_steps_new + 1))
paths_digital_25_up = np.zeros((NSimul, N_steps_new + 1))
paths_ko_120_up = np.zeros((NSimul, N_steps_new + 1))
paths_ko_150_up = np.zeros((NSimul, N_steps_new + 1))
paths_vanilla_call_up = np.zeros((NSimul, N_steps_new + 1))

paths_digital_10_down = np.zeros((NSimul, N_steps_new + 1))
paths_digital_25_down = np.zeros((NSimul, N_steps_new + 1))
paths_ko_120_down = np.zeros((NSimul, N_steps_new + 1))
paths_ko_150_down = np.zeros((NSimul, N_steps_new + 1))
paths_vanilla_call_down = np.zeros((NSimul, N_steps_new + 1))

# Set initial price at Forward Price with Shifted Rates
paths_digital_10_up[:, 0] = F_T1_up
paths_digital_25_up[:, 0] = F_T1_up
paths_ko_120_up[:, 0] = F_T1_up
paths_ko_150_up[:, 0] = F_T1_up
paths_vanilla_call_up[:, 0] = F_T1_up

paths_digital_10_down[:, 0] = F_T1_down
paths_digital_25_down[:, 0] = F_T1_down
paths_ko_120_down[:, 0] = F_T1_down
paths_ko_150_down[:, 0] = F_T1_down
paths_vanilla_call_down[:, 0] = F_T1_down

# Generate shocks for Monte Carlo
Z_up = np.random.standard_normal((NSimul, N_steps_new))
Z_down = np.random.standard_normal((NSimul, N_steps_new))

# Simulate paths with shifted rate & vol
simulate_paths(paths_digital_10_up, F_1y3y_up, vol_digital_10_up, div)
simulate_paths(paths_digital_25_up, F_1y3y_up, vol_digital_25_up, div)
simulate_paths(paths_ko_120_up, F_1y3y_up, vol_ko_120_up, div)
simulate_paths(paths_ko_150_up, F_1y3y_up, vol_ko_150_up, div)
simulate_paths(paths_vanilla_call_up, F_1y3y_up, vol_vanilla_call_up, div)

simulate_paths(paths_digital_10_down, F_1y3y_down, vol_digital_10_down, div)
simulate_paths(paths_digital_25_down, F_1y3y_down, vol_digital_25_down, div)
simulate_paths(paths_ko_120_down, F_1y3y_down, vol_ko_120_down, div)
simulate_paths(paths_ko_150_down, F_1y3y_down, vol_ko_150_down, div)
simulate_paths(paths_vanilla_call_down, F_1y3y_down, vol_vanilla_call_down, div)

# Compute final prices after (T-1) years for both scenarios
final_prices_digital_10_up = paths_digital_10_up[:, -1]
final_prices_digital_25_up = paths_digital_25_up[:, -1]
final_prices_ko_120_up = paths_ko_120_up[:, -1]
final_prices_ko_150_up = paths_ko_150_up[:, -1]
final_prices_vanilla_call_up = paths_vanilla_call_up[:, -1]

final_prices_digital_10_down = paths_digital_10_down[:, -1]
final_prices_digital_25_down = paths_digital_25_down[:, -1]
final_prices_ko_120_down = paths_ko_120_down[:, -1]
final_prices_ko_150_down = paths_ko_150_down[:, -1]
final_prices_vanilla_call_down = paths_vanilla_call_down[:, -1]

# Compute barrier hits
hit_10_up = (final_prices_digital_10_up >= Barrier_10).astype(int)
hit_25_up = (final_prices_digital_25_up >= Barrier_25).astype(int)
hit_10_down = (final_prices_digital_10_down >= Barrier_10).astype(int)
hit_25_down = (final_prices_digital_25_down >= Barrier_25).astype(int)

# Compute adjusted payoffs
digital_10_mc_up = np.where(hit_10_up == 1, 10, 0)
digital_10_mc_down = np.where(hit_10_down == 1, 10, 0)

digital_25_mc_up = np.where(hit_25_up == 1, 15, 0)
digital_25_mc_down = np.where(hit_25_down == 1, 15, 0)

call_ko_120_mc_up = np.where(hit_10_up == 0, np.maximum(Participation * (final_prices_ko_120_up - S0) / S0, 0) * 100, 0)
call_ko_120_mc_down = np.where(hit_10_down == 0, np.maximum(Participation * (final_prices_ko_120_down - S0) / S0, 0) * 100, 0)

call_ko_150_mc_up = np.where((hit_10_up == 1) & (hit_25_up == 0), np.maximum(Participation * (final_prices_ko_150_up - 1.2 * S0) / S0, 0) * 100, 0)
call_ko_150_mc_down = np.where((hit_10_down == 1) & (hit_25_down == 0), np.maximum(Participation * (final_prices_ko_150_down - 1.2 * S0) / S0, 0) * 100, 0)

vanilla_call_mc_up = np.maximum((final_prices_vanilla_call_up - 1.5 * S0) / S0, 0) * 100
vanilla_call_mc_down = np.maximum((final_prices_vanilla_call_down - 1.5 * S0) / S0, 0) * 100

# Compute expected payoffs
expected_payoff_digital_10_up = np.mean(digital_10_mc_up)
expected_payoff_digital_10_down = np.mean(digital_10_mc_down)

expected_payoff_digital_25_up = np.mean(digital_25_mc_up)
expected_payoff_digital_25_down = np.mean(digital_25_mc_down)

expected_payoff_call_ko_120_up = np.mean(call_ko_120_mc_up)
expected_payoff_call_ko_120_down = np.mean(call_ko_120_mc_down)

expected_payoff_call_ko_150_up = np.mean(call_ko_150_mc_up)
expected_payoff_call_ko_150_down = np.mean(call_ko_150_mc_down)

expected_payoff_vanilla_call_up = np.mean(vanilla_call_mc_up)
expected_payoff_vanilla_call_down = np.mean(vanilla_call_mc_down)

# Compute new discount factors with shifted rates
discount_factor_options_up = np.exp(-F_1y3y_up * (T-2))
discount_factor_options_down = np.exp(-F_1y3y_down * (T-2))

discount_factor_zcb_up = np.exp(-F_1y3y_up * (T-1))
discount_factor_zcb_down = np.exp(-F_1y3y_down * (T-1))

# Adjusted zero-coupon bond price
zero_coupon_bond_price_up = discount_factor_zcb_up * 100
zero_coupon_bond_price_down = discount_factor_zcb_down * 100

# Adjusted prices for options and structured product components
price_digital_10_up = expected_payoff_digital_10_up * discount_factor_options_up
price_digital_10_down = expected_payoff_digital_10_down * discount_factor_options_down

price_digital_25_up = expected_payoff_digital_25_up * discount_factor_options_up
price_digital_25_down = expected_payoff_digital_25_down * discount_factor_options_down

price_call_ko_120_up = expected_payoff_call_ko_120_up * discount_factor_options_up * Participation
price_call_ko_120_down = expected_payoff_call_ko_120_down * discount_factor_options_down * Participation

price_call_ko_150_up = expected_payoff_call_ko_150_up * discount_factor_options_up * Participation
price_call_ko_150_down = expected_payoff_call_ko_150_down * discount_factor_options_down * Participation

price_vanilla_call_up = expected_payoff_vanilla_call_up * discount_factor_options_up * Participation
price_vanilla_call_down = expected_payoff_vanilla_call_down * discount_factor_options_down * Participation

# Compute total structured product price under different scenarios
final_product_price_up = (
    price_call_ko_120_up +
    price_call_ko_150_up +
    price_digital_10_up +
    price_digital_25_up +
    price_vanilla_call_up
) + zero_coupon_bond_price_up

final_product_price_down = (
    price_call_ko_120_down +
    price_call_ko_150_down +
    price_digital_10_down +
    price_digital_25_down +
    price_vanilla_call_down
) + zero_coupon_bond_price_down

# Compute bid-ask prices
P_bid = final_product_price_down
P_ask = final_product_price_up

st.markdown("---")

# Create DataFrame for initial vs adjusted pricing
df_pricing_evolution = pd.DataFrame({
    "Component": [
        "One-Touch Digital (120%)",
        "One-Touch Digital (150%)",
        "Knock-Out Call (100%, KO at 120%)",
        "Knock-Out Call (120%, KO at 150%)",
        "Vanilla Call (150%)",
        "Zero Coupon Bond",
        "Total"
    ],
    "Initial Price (%)": [
        price_digital_10,
        price_digital_25,
        price_call_ko_120,
        price_call_ko_150,
        price_vanilla_call,
        zero_coupon_bond_price,
        final_product_price,
    ],
    "Price in 1 year (%)": [
        price_digital_10_new_2,
        price_digital_25_new_2,
        price_call_ko_120_new_2,
        price_call_ko_150_new_2,
        price_vanilla_call_new_2,
        zero_coupon_bond_price_new_2,
        final_product_price_new_2,
    ],
    "Adjusted Price (Bid)": [
        price_digital_10_down,
        price_digital_25_down,
        price_call_ko_120_down,
        price_call_ko_150_down,
        price_vanilla_call_down,
        zero_coupon_bond_price_down,
        final_product_price_down,
    ],
    "Adjusted Price (Ask)": [
        price_digital_10_up,
        price_digital_25_up,
        price_call_ko_120_up,
        price_call_ko_150_up,
        price_vanilla_call_up,
        zero_coupon_bond_price_up,
        final_product_price_up,
    ]
})

df_pricing_evolution.set_index("Component", inplace=True)
df_pricing_evolution = df_pricing_evolution.apply(pd.to_numeric, errors="coerce")

# Compute Total Product Prices for each scenario
total_initial_price = df_pricing_evolution["Initial Price (%)"].sum()-final_product_price
total_price_new_2 = df_pricing_evolution["Price in 1 year (%)"].sum()-final_product_price_new_2
total_price_bid = df_pricing_evolution["Adjusted Price (Bid)"].sum()-final_product_price_down
total_price_ask = df_pricing_evolution["Adjusted Price (Ask)"].sum()-final_product_price_up

col1, col2 = st.columns([3, 1])

# Component Breakdown & Evolution
with col1:
    st.markdown("<div style='font-size:20px; font-weight:bold;'>📊 Component Breakdown & Evolution</div>", unsafe_allow_html=True)
    st.dataframe(df_pricing_evolution.style.format("{:.2f}%"), use_container_width=True)

with col2:
    # Title
    st.markdown("<div style='font-size:20px; font-weight:bold;'>💰 Product Prices</div>", unsafe_allow_html=True)

    # Price in 1 Year
    st.markdown(f"""
        <div style="padding: 1px 1px; min-width: 60px; border-radius: 4px; 
                    border: 1px solid black; background-color: #f7f7f7; text-align: center;
                    box-shadow: 1px 1px 1px rgba(0, 0, 0, 0.1);">
            <h4 style="color: black; font-size: 16px; margin: 0px 0; line-height: 1.2;">Price in 1 Year</h4>
            <h3 style="color: black; font-size: 18px; margin: 0px 0;"><strong>{total_price_new_2:.2f}%</strong></h3>
        </div>
    """, unsafe_allow_html=True)

    # Bid Price
    color_bid = "#D90000" if total_price_bid < 100 else "#008000"
    st.markdown(f"""
        <div style="padding: 1px 1px; min-width: 60px; border-radius: 4px; 
                    border: 1px solid {color_bid}; background-color: #f7f7f7; text-align: center;
                    box-shadow: 1px 1px 1px rgba(0, 0, 0, 0.1);">
            <h4 style="color: {color_bid}; font-size: 16px; margin: 0px 0; line-height: 1.2;">Bid Price</h4>
            <h3 style="color: {color_bid}; font-size: 18px; margin: 0px 0;"><strong> {total_price_bid:.2f}%</strong></h3>
        </div>
    """, unsafe_allow_html=True)

    # Ask Price
    color_ask = "#008000" if total_price_ask > 100 else "#D90000"
    st.markdown(f"""
        <div style="padding: 1px 1px; min-width: 60px; border-radius: 4px; 
                    border: 1px solid {color_ask}; background-color: #f7f7f7; text-align: center;
                    box-shadow: 1px 1px 1px rgba(0, 0, 0, 0.1);">
            <h4 style="color: {color_ask}; font-size: 16px; margin: 0px 0; line-height: 1.2;">Ask Price</h4>
            <h3 style="color: {color_ask}; font-size: 18px; margin: 0px 0;"><strong> {total_price_ask:.2f}%</strong></h3>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("## 💡 Investor Recommendation: Should You Hold or Sell?")

# Explanation
st.markdown("""
Your investment started at **100%** of the nominal value. Now, based on market movements, the **current bid price** reflects how much you would get if you sold your structured product today.
""")

if P_bid > 100:
    st.success(f"✅ **Sell the product** – The market bid price is **above 100%**, meaning you can exit with a profit.")
elif P_bid < 100:
    st.error(f"❌ **Hold the product** – The market bid price is **below 100%**, meaning selling now would result in a loss.")
else:
    st.warning(f"⚖ **Neutral Decision** – The market bid price is exactly **100%**, so there is no financial gain or loss.")

st.metric(
    label="📊 **Current Bid Price**",
    value=f"{P_bid:.2f}%",
    delta=f"{P_bid - 100:.2f}%",
    delta_color="inverse"  
)

st.markdown("""
💡 **Key Takeaways:**  
- If **P_bid > 100%**, selling is advantageous as you are making a profit.  
- If **P_bid < 100%**, holding could be preferable to avoid selling at a loss.  
""")



