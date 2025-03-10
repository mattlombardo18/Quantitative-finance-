#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:07:34 2025

@author: matthieulombardo
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
from scipy.stats import norm



def black_scholes(S,K,T,r,sigma,option_type="call"):
  d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
  elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
  else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
  return price
 
    
# Greeks

def calculate_greeks(S, K, T, r, sigma):
    d1 = (np.log(S/K)+(r+0.5 * sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) # Delta
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T) ) # Gamma
    vega = S * norm.pdf(d1) * np.sqrt(T) # Vega
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)# Theta
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) # Rho 
    return delta, gamma, vega, theta, rho

# Streamlit App
st.title("Options Pricing Dashboard")

# Input Parameters
st.sidebar.header("Options Parameters")
spot_price = st.sidebar.slider("Spot Price (S)", 50, 150, 100)
strike_price = st.sidebar.slider("Strike Price (K)", 50, 150, 100)
time_to_maturity = st.sidebar.slider("Time to Maturity (T, years)", 0.1, 2.0, 1.0, 0.1)
risk_free_rate = st.sidebar.slider("Risk-Free Rate (r)", 0.0, 0.1, 0.05, 0.01)
volatility = st.sidebar.slider("Volatility (σ)", 0.1, 1.0, 0.2, 0.01)
option_type = st.sidebar.radio("Option Type", ["Call", "Put"])

# Calculate Option Price
price = black_scholes(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type.lower())
st.write(f"## {option_type} Option Price (Black-Scholes): ${price:.2f}")

# Greeks
delta, gamma, vega, theta, rho = calculate_greeks(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)
st.write("### Greeks")
st.write(f"**Delta**: {delta:.2f}")
st.write(f"**Gamma**: {gamma:.2f}")
st.write(f"**Vega**: {vega:.2f}")
st.write(f"**Theta**: {theta:.2f}")
st.write(f"**Rho**: {rho:.2f}")

# Payoff Diagram
st.write("### Payoff Diagram")
spot_prices = np.linspace(50, 150, 100)
if option_type == "Call":
    payoff = np.maximum(spot_prices - strike_price, 0)
else:
    payoff = np.maximum(strike_price - spot_prices, 0)

plt.figure(figsize=(10, 5))
plt.plot(spot_prices, payoff, label="Payoff")
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Spot Price at Expiration")
plt.ylabel("Payoff")
plt.title(f"{option_type} Option Payoff Diagram")
plt.legend()
st.pyplot(plt)
plt.close()

# Volatility Surface
st.write("### Volatility Surface")
maturities = np.linspace(0.1, 2, 10)
strikes = np.linspace(50, 150, 50)
X, Y = np.meshgrid(strikes, maturities)
Z = volatility * (1 + 0.1 * np.sin((X - spot_price) / 20) + 0.1 * np.cos(Y * 2))

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel("Strike Price")
ax.set_ylabel("Maturity (Years)")
ax.set_zlabel("Implied Volatility")
ax.set_title("Volatility Surface")
st.pyplot(fig)
plt.close()

# Greeks Sensitivity Analysis
st.write("### Greeks Sensitivity")
spot_prices = np.linspace(50, 150, 100)
deltas = [calculate_greeks(spot, strike_price, time_to_maturity, risk_free_rate, volatility)[0] for spot in spot_prices]
gammas = [calculate_greeks(spot, strike_price, time_to_maturity, risk_free_rate, volatility)[1] for spot in spot_prices]

# Plot Delta Sensitivity
plt.figure(figsize=(10, 5))
plt.plot(spot_prices, deltas, label="Delta")
plt.xlabel("Spot Price")
plt.ylabel("Delta")
plt.title("Delta Sensitivity to Spot Price")
plt.legend()
st.pyplot(plt)
plt.close()

# Plot Gamma Sensitivity
plt.figure(figsize=(10, 5))
plt.plot(spot_prices, gammas, label="Gamma", color='orange')
plt.xlabel("Spot Price")
plt.ylabel("Gamma")
plt.title("Gamma Sensitivity to Spot Price")
plt.legend()
st.pyplot(plt)
plt.close()

def monte_carlo_option_pricing_full_path(S, K, T, r, sigma, option_type, num_simulations=10000, num_steps=252):
    np.random.seed(42)
    dt = T / num_steps  # Step size
    prices = np.zeros((num_simulations, num_steps + 1))
    prices[:, 0] = S  # Initialize prices with the initial spot price

    for t in range(1, num_steps + 1):
        # Generate random shocks for each step
        z = np.random.randn(num_simulations)
        prices[:, t] = prices[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    # Extract the price at maturity
    final_prices = prices[:, -1]

    # Calculate payoff
    if option_type == "call":
        payoffs = np.maximum(final_prices - K, 0)
    else:
        payoffs = np.maximum(K - final_prices, 0)

    # Discount payoffs back to present value
    return np.exp(-r * T) * np.mean(payoffs)

mc_price = monte_carlo_option_pricing_full_path(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type.lower())
st.write(f"### Monte Carlo Estimated Price: ${mc_price:.2f}")


# Monte Carlo Paths
def monte_carlo_paths(S, T, r, sigma, num_simulations=100, num_steps=252):
    dt = T / num_steps
    paths = np.zeros((num_simulations, num_steps + 1))
    paths[:, 0] = S  # Initial spot price for all paths

    for t in range(1, num_steps + 1):
        z = np.random.randn(num_simulations)  # Random shocks
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    return paths

st.write("### Monte Carlo Simulated Paths")

# Simulate paths
num_simulations = 10  # Number of paths to plot (to keep it visually clear)
num_steps = 252  # Daily steps for one year
paths = monte_carlo_paths(spot_price, time_to_maturity, risk_free_rate, volatility, num_simulations, num_steps)

# Plot the paths
plt.figure(figsize=(10, 5))
time = np.linspace(0, time_to_maturity, num_steps + 1)
for i in range(num_simulations):
    plt.plot(time, paths[i], lw=1)

plt.xlabel("Time to Maturity (Years)")
plt.ylabel("Price")
plt.title("Monte Carlo Simulated Paths for Underlying Asset")
plt.grid(True)
st.pyplot(plt)
plt.close()

# 📌 Footer
st.markdown("""
    ---
    📌 **Black & Scholes Pricer** | Created by [Matthieu Lombardo](https://www.linkedin.com/in/matthieu-lombardo)
""", unsafe_allow_html=True)
