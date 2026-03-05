import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from analysis import run_simulation_logic  # Ensure this function is in analysis.py

st.set_page_config(page_title="Stratification Simulator", layout="wide")

st.title("📊 Social Stratification Simulator")
st.markdown("Exploring intergenerational mobility through Agent-Based Modeling.")

# Sidebar for Parameters
st.sidebar.header("Simulation Settings")
scenario = st.sidebar.selectbox("Macroeconomic Scenario", 
                                ["United States", "Finland", "Turkey", "Rigid", "Hyper-Mobile"])
agents = st.sidebar.slider("Number of Agents", 100, 1000, 500)
epochs = st.sidebar.slider("Generations", 1, 10, 5)

if st.sidebar.button("Run Simulation"):
    with st.spinner("Calculating socioeconomic transitions..."):
        # This calls your existing logic
        matrix = run_simulation_logic(scenario=scenario, num_agents=agents, generations=epochs)
        
        # Display Results
        st.subheader(f"Transition Matrix: {scenario}")
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
        ax.set_xlabel("Descendant Quintile")
        ax.set_ylabel("Parent Quintile")
        st.pyplot(fig)