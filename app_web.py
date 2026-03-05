import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from analysis import run_mobility_simulation

# Parameter configurations matching the GUI
PARAM_INFO = {
    "num_agents": ("Number of Agents", "Total population size in the simulation"),
    "width": ("Grid Width", "Horizontal size of the urban territory"),
    "height": ("Grid Height", "Vertical size of the urban territory"),
    "tax_rate": ("Income Tax Rate", "Percentage of income collected for the UBI pool"),
    "tax_inheritance": ("Estate Tax Rate", "Tax on parent's capital during generational turnover"),
    "c_min": ("Subsistence Threshold", "Minimum wealth level protected from housing costs"),
    "base_wage": ("Base Wage", "Baseline multiplier in the Cobb-Douglas income function"),
    "interest_rate": ("Interest Rate", "Compounding growth rate on stored economic capital"),
    "h_base": ("Public Education Endowment", "Universal education endowment granted at birth"),
    "omega_h": ("HC Inheritance Rate", "Parent-to-child transmission rate of human capital"),
    "h_variance": ("Talent Variance", "Std. deviation of stochastic shock on inherited HC"),
    "delta_h": ("HC Depreciation Rate", "Rate at which human capital decays over time"),
    "theta": ("Education Growth Rate", "Baseline multiplier for human capital growth"),
    "eta": ("Investment Efficiency", "How effectively wealth converts into HC growth"),
    "mu": ("Base Rent Multiplier", "Fundamental cost of living before density adjustments"),
    "kappa": ("Density Elasticity", "How neighborhood crowding inflates local rent"),
    "rho": ("Housing Income Share", "Proportion of income allocated toward housing"),
    "phi": ("Housing Wealth Share", "Proportion of surplus wealth allocated toward housing"),
    "lambda_1": ("Neighborhood Quality Weight", "Importance of quality in relocation decisions"),
    "gamma_distance": ("Spatial Friction", "Distance penalty in network tie formation"),
    "lambda_2": ("Homophily Penalty", "Class distance penalty in relocation utility"),
    "alpha": ("HC Output Elasticity", "Human capital exponent in income function"),
    "beta": ("SC Output Elasticity", "Social capital exponent in income function"),
    "weight_e": ("Network Econ. Weight", "Weight of peers' economic capital in social capital"),
    "weight_h": ("Network HC Weight", "Weight of peers' human capital in social capital"),
    "rho_net": ("Network Tie Retention", "Probability ties survive generational turnover"),
    "max_lifespan": ("Generational Lifespan", "Ticks per demographic epoch"),
    "generations": ("Number of Generations", "Total demographic epochs to simulate"),
}

SCENARIO_DESCRIPTIONS = {
    "Rigid Imaginary": "A hyper-stratified dystopian baseline with high educational transmission, extreme spatial friction, and severe homophily. Social position is almost entirely determined by birth.",
    "United States": "Models moderate-to-high intergenerational persistence with average housing market elasticity, moderate spatial divides, and a 40% estate tax.",
    "Turkey": "Simulates high persistence and strong elite closure characterized by significant educational inequality, low effective estate taxes (15%), and strong spatial segregation.",
    "Finland": "The Nordic welfare model with high redistributive taxation (45%), massive public education endowments, minimal homophily, and Universal Basic Income.",
    "Hyper-Mobile Imaginary": "Perfect equality of opportunity: 99% estate tax, zero spatial friction, and outcomes driven entirely by variance rather than parental baseline.",
}

params_config = {
    "General": ["num_agents", "width", "height"],
    "Macroeconomic": ["tax_rate", "tax_inheritance", "c_min", "base_wage", "interest_rate"],
    "Education": ["h_base", "omega_h", "h_variance", "delta_h", "theta", "eta"],
    "Urban & Housing": ["mu", "kappa", "rho", "phi", "lambda_1"],
    "Social Network": ["gamma_distance", "lambda_2", "alpha", "beta", "weight_e", "weight_h", "rho_net"],
    "Temporal": ["max_lifespan", "generations"]
}

default_params = {
    "num_agents": 500, "width": 20, "height": 20,
    "tax_rate": 0.25, "tax_inheritance": 0.40, "c_min": 5.0,
    "base_wage": 10.0, "interest_rate": 0.02, "h_base": 2.0, "omega_h": 0.4,
    "h_variance": 2.0, "delta_h": 0.05, "theta": 0.1, "eta": 0.01,
    "mu": 15.0, "kappa": 0.1, "rho": 0.3, "phi": 0.05,
    "lambda_1": 2.0, "gamma_distance": 0.2, "lambda_2": 0.05,
    "alpha": 0.3, "beta": 0.2, "weight_e": 0.5, "weight_h": 0.5,
    "rho_net": 0.7, "max_lifespan": 50, "generations": 5
}

st.set_page_config(page_title="Stratification Simulator", layout="wide")

st.title("📊 Social Stratification Simulator")
st.markdown("Agent-based intergenerational mobility & spatial segregation model.")

if st.button("Read Me / Project Information"):
    st.info("The Social Stratification Simulator uses an agent-based approach to explore intergenerational income mobility, educational gradients, spatial segregation, and network effects under different macroeconomic topologies. You can adjust detailed scenario parameters via the sidebar to observe varying social persistence outcomes.")

# Sidebar for Parameters
st.sidebar.header("Simulation Settings")

scenario = st.sidebar.selectbox("Macroeconomic Scenario", 
                                list(SCENARIO_DESCRIPTIONS.keys()),
                                index=1)
st.sidebar.caption(SCENARIO_DESCRIPTIONS[scenario])

st.sidebar.subheader("Detailed Parameters")

# Initialize kwargs for custom parameters
kwargs = {}

# Use expanders to match GUI parameter categories
for cat, params in params_config.items():
    with st.sidebar.expander(cat, expanded=False):
        for p in params:
            display_name, desc = PARAM_INFO.get(p, (p, ""))
            # Distinguish integers and floats for proper input type
            default_val = default_params.get(p, 0.0)
            
            # Show description via Streamlit help tooltip
            if isinstance(default_val, int):
                val = st.number_input(display_name, value=default_val, help=desc, step=1, key=p)
            else:
                val = st.number_input(display_name, value=float(default_val), help=desc, step=0.01, format="%.2f", key=p)
            kwargs[p] = val

agents = kwargs.pop("num_agents")
epochs = kwargs.pop("generations")
width = kwargs.pop("width")
height = kwargs.pop("height")

if st.button("Run Simulation", type="primary"):
    with st.spinner("Calculating socioeconomic transitions..."):
        # This calls logic in analysis.py
        matrix = run_mobility_simulation(
            scenario=scenario, 
            num_agents=agents, 
            generations=epochs,
            width=width,
            height=height,
            **kwargs
        )

        st.subheader(f"Transition Matrix: {scenario}")
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
        
        ax.set_xlabel("Descendant Wealth Quintile")
        ax.set_ylabel("Parent Wealth Quintile")
        ax.invert_yaxis()
        
        st.pyplot(fig)
