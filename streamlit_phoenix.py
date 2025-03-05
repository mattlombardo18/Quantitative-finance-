#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 12:44:34 2025

@author: matthieulombardo
"""
import subprocess
import sys

def install_if_needed(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Liste des packages à vérifier et installer si nécessaire
packages = {
    "streamlit": "streamlit",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "pandas": "pandas"
}

for module, package in packages.items():
    install_if_needed(package)
    
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.stats import norm
import pandas as pd

# Streamlit Interface
st.title("Phoenix Autocall Dashboard")
st.sidebar.header("Autocall Parameters")

np.random.seed(42)  # Répétabilité des résultats

# Paramètres interactifs
initial_index_level = st.sidebar.slider("Initial Index Level", 50, 200, 150)
autocall_trigger_level = 1.10 * initial_index_level
coupon_barrier_level = 0.7 * initial_index_level
protection_barrier_level = 0.6 * initial_index_level
coupon_rate = st.sidebar.slider("Coupon Rate", 0.0, 0.1, 0.05, 0.01)
num_simulation = st.sidebar.slider("Number of Simulations", 100, 1000, 500)
num_observation = 8
risk_free_rate = st.sidebar.slider("Risk-Free Rate (r)", 0.0, 0.1, 0.05, 0.01)
volatility = st.sidebar.slider("Volatility (σ)", 0.1, 1.0, 0.2, 0.01)

# Simuler les trajectoires de l'indice
dt = 1 / 252  # Jour de marché
time_steps = 252 * 2  # Deux ans
index_paths = np.zeros((num_simulation, time_steps + 1))
index_paths[:, 0] = initial_index_level

for i in range(1, time_steps + 1):
    z = np.random.standard_normal(num_simulation)
    index_paths[:, i] = index_paths[:, i - 1] * np.exp((risk_free_rate - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z)

# Dates d'observation
observation_dates = np.arange(63, time_steps + 1, 63)

# Calcul des payoffs
payoff = np.zeros(num_simulation)
autocalled = np.zeros(num_simulation, dtype=bool)

for j in range(num_simulation):
    for k in range(num_observation):
        if index_paths[j, observation_dates[k]] >= autocall_trigger_level:
            autocalled[j] = True
            payoff[j] = 100 * (1 + coupon_rate * (k + 1))
            break
        elif index_paths[j, observation_dates[k]] >= coupon_barrier_level:
            payoff[j] += coupon_rate * 100
    if not autocalled[j]:
        if index_paths[j, -1] >= protection_barrier_level:
            payoff[j] += 100
        else:
            payoff[j] += index_paths[j, -1]

# Afficher les résultats
st.write(f"Payoff (première simulation) : {payoff[0]:.2f}")

# Actualiser les payoffs
discounted_payoff = np.exp(-risk_free_rate * 2) * payoff
st.write(f"Discounted Payoff (première simulation) : {discounted_payoff[0]:.2f}")

# Calcul du prix
note_price = np.mean(discounted_payoff)
st.write(f"Price of the Phoenix autocall note: {note_price:.2f}")

# Histogramme des payoffs
st.write("### Distribution of Payoffs")
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(payoff, bins=50, alpha=0.75, color='blue', edgecolor='black')
ax.set_xlabel('Payoff')
ax.set_ylabel('Frequency (log)')
ax.set_title('Distribution of Payoffs')
ax.grid(True)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.axvline(x=note_price, color='red', linestyle='--', linewidth=2, label='Note Price: ' + f"{note_price:.2f}")
ax.legend()
st.pyplot(fig)

# Calcul des Greeks
def calculate_greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
             r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100

    return delta, gamma, vega, theta, rho

# Visualisation des Greeks
st.write("### Greeks Analysis")
T = 2  # Temps jusqu'à expiration en années
greeks = {"Spot Price": [], "Delta": [], "Gamma": [], "Vega": [], "Theta": [], "Rho": []}
spot_prices = np.linspace(initial_index_level * 0.5, initial_index_level * 1.5, 100)

for S in spot_prices:
    delta, gamma, vega, theta, rho = calculate_greeks(S, initial_index_level, T, risk_free_rate, volatility)
    greeks["Spot Price"].append(S)
    greeks["Delta"].append(delta)
    greeks["Gamma"].append(gamma)
    greeks["Vega"].append(vega)
    greeks["Theta"].append(theta)
    greeks["Rho"].append(rho)

greeks_df = pd.DataFrame(greeks)
st.line_chart(greeks_df.set_index("Spot Price"))

# Fonction pour tracer un chemin simulé avec annotations
def plot_path(idx, observation_dates):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(index_paths[idx], label='Simulated Index Path')

    # Ajouter les niveaux clés
    ax.axhline(y=autocall_trigger_level, color='g', linestyle='--', label='Autocall Trigger Level')
    ax.axhline(y=coupon_barrier_level, color='r', linestyle='--', label='Coupon Barrier Level')
    ax.axhline(y=protection_barrier_level, color='b', linestyle='--', label='Protection Barrier Level')

    # Ajouter les annotations
    autocall_touched = False
    coupon_touched = False
    protection_touched = False

    for i, level in enumerate(index_paths[idx]):
        if level >= autocall_trigger_level and not autocall_touched:
            ax.axvline(x=i, color='g', linestyle='--', linewidth=2)
            ax.text(i + 5, autocall_trigger_level - 5, 'AUTOCALL HIT', color='g', fontsize=10, rotation=90)
            autocall_touched = True
        elif level <= coupon_barrier_level and not coupon_touched:
            ax.axvline(x=i, color='r', linestyle='--', linewidth=2)
            ax.text(i + 5, coupon_barrier_level + 5, 'COUPON BARRIER HIT', color='r', fontsize=10, rotation=90)
            coupon_touched = True
        elif level <= protection_barrier_level and not protection_touched:
            ax.axvline(x=i, color='b', linestyle='--', linewidth=2)
            ax.text(i + 5, protection_barrier_level + 5, 'PROTECTION BARRIER HIT', color='b', fontsize=10, rotation=90)
            protection_touched = True

    # Dates d'observation
    for i, date in enumerate(observation_dates):
        if i == 0:
            ax.axvline(x=date, color='black', linestyle='--', linewidth=1, label="Observation Date")
        else:
            ax.axvline(x=date, color='black', linestyle='--', linewidth=1)

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Index Level')
    ax.set_title('Simulated Index Path with Key Levels')
    ax.legend()
    ax.grid(True)
    return fig

# Tracer un chemin simulé
st.write("### Simulated Index Path")
selected_path = st.sidebar.slider("Select Simulation Path", 0, num_simulation - 1, 24)
fig = plot_path(selected_path, observation_dates)
st.pyplot(fig)
