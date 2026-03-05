# Macro-Sociological Stratification and Mobility Simulator

This repository contains an Agent-Based Model (ABM) developed in Python utilizing the Mesa framework. The simulation models the emergence of social stratification, spatial segregation, and intergenerational mobility across multiple demographic epochs. By defining micro-level structural constraints, the application observes how inequality compounds and reproduces over time.

## Theoretical Framework

The simulation architecture operationalizes several foundational sociological and economic theories to generate realistic macroeconomic dynamics.

1.  **Multidimensional Capital:** Grounded in Pierre Bourdieu's theory of capital, agents accumulate three distinct resources. Economic capital represents liquid wealth. Human capital represents education and institutional credentials. Social capital represents the economic utility of an agent's network ties.
2.  **Spatial Sorting and Bid-Rent Theory:** Agents exist within an urban grid featuring varying institutional quality. The housing market logic forces agents to evaluate their budget constraint against dynamic spatial costs. High-demand, high-quality nodes increase in price, systematically evicting low-capital agents to peripheral zones.
3.  **Network Homophily:** Social networks form based on a gravity model. Agents calculate the probability of forming a tie using spatial distance and class distance. This mechanism simulates elite closure and resource hoarding.
4.  **Intergenerational Mobility:** The simulation tracks the transition of capital across biological generations. Offspring inherit a percentage of their parent's economic wealth, a baseline of their human capital (subject to stochastic variance), and fragments of their social network. The model generates transition matrices to quantify social reproduction, comparable to empirical methodologies utilized in contemporary urban studies and economics.

## Mathematical Formalization

The simulation relies on specific mathematical functions to drive agent behavior and capital accumulation. 

### Income Generation Function
Agents generate income (Y) using a modified Cobb-Douglas production function, converting human and social inputs into economic output:

Y = A * (C_H)^alpha * (C_S)^beta * epsilon

Where A is the base wage, alpha and beta are the output elasticities of human and social capital, and epsilon is a stochastic shock variable representing market variance.

### Network Tie Formation
The probability (p) of two agents forming a network tie incorporates both spatial friction and homophily:

p = 1 / (1 + exp(gamma_1 * d + gamma_2 * |C_i - C_j| - gamma_0))

Where d is the Euclidean spatial distance, C is the composite class index of the agent, gamma_1 dictates spatial friction, and gamma_2 dictates homophily preference.

## System Architecture

The codebase is modularized to separate micro-level agent logic from macro-level environmental parameters and analytical extraction routines.

* `agents.py`: Contains the `StratificationAgent` class. This module defines all micro-level behaviors, including income generation, tax calculation, housing evaluation, and the intergenerational transfer of capital.
* `model.py`: Contains the `UrbanEnvironment` class. This module acts as the macro-environment, managing the spatial grid, network graph, dynamic housing costs, and the Universal Basic Income (UBI) redistribution mechanism. It also houses the scenario configuration parameters.
* `metrics.py`: Isolates the mathematical functions required by the Mesa DataCollector. It includes the algorithms to calculate the Gini coefficient and Moran's I for spatial segregation.
* `analysis.py`: Manages the execution loops and data extraction. It converts longitudinal agent data into Pandas DataFrames and computes the intergenerational mobility probability matrix.
* `main.py`: The primary entry point. It provides an interactive command-line interface (CLI) to select macroeconomic scenarios and triggers the visualization routines.

## Macroeconomic Scenarios

The `UrbanEnvironment` class accepts a scenario parameter to calibrate the structural constraints, allowing the simulation to approximate specific empirical realities.

1.  **Rigid Imaginary:** A hyper-stratified, dystopian baseline characterized by high parent-to-child educational transmission, high spatial friction, and severe homophily.
2.  **United States:** Models moderate-to-high intergenerational persistence with average housing market elasticity and moderate spatial divides.
3.  **Turkey:** Simulates high persistence and strong elite closure, characterized by significant educational inequality, low effective estate taxes, and strong spatial segregation.
4.  **Finland:** Represents the Nordic welfare model. It features a high redistributive tax rate, massive public education endowments, low homophily, and a Universal Basic Income.
5.  **Hyper-Mobile Imaginary:** A theoretical model of perfect equality of opportunity, featuring a 99% estate tax, zero spatial friction, and outcomes driven entirely by variance rather than parental baseline.

## What It Simulates

The UrbanEnvironment class functions as the macro-sociological framework within our simulation. It acts as the structural container that enforces physical, economic, and social constraints on the individual agents, dictating the friction and opportunities they encounter.

Functionally, it simulates the systemic forces of a city or nation state. By adjusting its hyperparameters, you fundamentally alter the society's economic model and spatial dynamics.

Here is a rigorous breakdown of its core components and what they represent sociologically.

1. Spatial Infrastructure and Institutional Quality
The environment utilizes a spatial grid (mesa.space.MultiGrid) to represent urban geography. Each coordinate on this grid contains an intrinsic neighborhood_quality score assigned during initialization. This array simulates the uneven distribution of public goods, such as municipal infrastructure, safety, and school district quality. An agent's location directly bounds its capacity to develop human capital, operationalizing the geographic lottery of birth.

2. The Dynamic Housing Market
The environment actively simulates urban economic geography via the get_dynamic_spatial_cost method. It calculates localized living expenses by applying a bid-rent mechanism. The base rent of a neighborhood is multiplied by a density elasticity parameter (kappa). As high-quality cells attract more agents, the localized demand drives up the cost of living. This component serves as the primary engine for spatial segregation, systematically pricing out low-capital agents and forcing them into under-resourced peripheries.

3. Social Network Architecture
The environment maintains a continuous social graph using networkx.Graph to track relationships. Through the update_networks method, the macro-environment dictates how social capital forms by applying a gravity model of interaction. It evaluates the spatial distance and the absolute socioeconomic distance between any two agents. By tuning the spatial friction (gamma_distance) and homophily (lambda_2) parameters, this component simulates phenomena ranging from highly integrated civic communities to severe elite closure and class isolation.

4. Macroeconomic Policy and The Welfare State
The environment manages systemic fiscal policy. It collects the taxes generated by individual agents into a central municipal treasury (tax_pool). The distribute_welfare method then reallocates this wealth evenly across the population at the end of each temporal cycle. This component allows the simulation to test the efficacy of redistributive mechanisms, ranging from entirely laissez-faire capitalism (where the tax rate is zero) to a robust Nordic welfare state featuring a Universal Basic Income.

5. Demographic Turnover and Social Reproduction
The environment acts as the biological clock for the population. It enforces a maximum generational lifespan (max_lifespan), triggering simultaneous inheritance events. During these generational turnovers, the environment imposes macro-level inheritance taxes (tax_inheritance) and provides a baseline public education endowment (h_base) to the offspring. This component controls the mathematical degree to which parent wealth determines descendant outcomes.

6. The Analytical Engine
The environment houses a data collection module. At the end of every computational tick, it executes complex sociological metrics to quantify the systemic state. It extracts the Gini coefficient to measure economic inequality and calculates a localized Moran's I to measure the spatial autocorrelation of wealth (geographic segregation).


## Five Epochs as Default

An epoch is the fundamental chronological unit of social reproduction in the model. It translates to one complete demographic generation here. 

When the application states it is "simulating across 5 epochs," it means it is tracking the accumulation and transfer of capital across five distinct generations of agents.

Here is exactly how an epoch functions mathematically and chronologically within the architecture we built:

- The Lifespan (The Ticks): One epoch consists of a specific number of computational time steps (or "ticks"). In your UrbanEnvironment, this is defined by the max_lifespan parameter, which is set to 50. Therefore, one epoch equals 50 ticks of the simulation loop.

- Intra-Epoch Mechanics (Accumulation): During these 50 ticks, the agents go through their life cycle. They work to generate income, pay rent in the spatial housing market, pay taxes, invest in their human capital, and form social network ties.

- The Epoch Boundary (Inheritance): When an agent reaches the end of the epoch (age >= self.model.max_lifespan), the execute_inheritance() method is triggered. The agent "dies" and is immediately replaced by their descendant. The simulation calculates the estate tax, applies human capital variance (luck/talent), and transfers the remaining wealth and social ties to this new generation.

- The Analytical Output: To calculate the final intergenerational mobility matrix, the analysis.py module takes a snapshot of the agents' wealth at Epoch 1 (the initial parents) and compares it to their descendants' wealth at the end of Epoch 5 (the great-great-grandchildren).


## The Sociological Takeaway

Despite the massive UBI payouts, the bottom quintile retention rate barely flinched, resting at an ironclad 87%. Why did the famous Nordic welfare model fail to save our poorest agents in this environment?

The simulation mathematically demonstrates Universal Inflation and the Rent Trap.

In our UrbanEnvironment, housing costs are dynamically calculated based on neighborhood quality and density (demand). When we gave every single agent a UBI payout, we increased the baseline purchasing power of the entire population simultaneously.

Because the spatial grid is constrained, this universal influx of cash simply drove up the localized cost of living. The poorest agents received their welfare checks, but those checks were immediately absorbed by the dynamic housing market. They grew richer in absolute terms (they likely have more total capital than the poor agents in the Turkey scenario), but because quintiles measure relative wealth, they remained firmly stuck in the bottom 20% compared to the rest of society.

This is exactly why sociologists and urban economists argue that UBI alone cannot fix stratification if it is deployed within an unregulated housing market. If you subsidize income without expanding housing supply or implementing rent controls, the welfare money simply trickles up to property owners.

To actually achieve the high mobility rates seen in real-world Nordic countries, the macro-environment requires decommodified housing (e.g., massive public housing blocks that are immune to dynamic market pricing).

## 

The Hyper-Mobile Imaginary scenario pushes the simulation's parameters to their theoretical limits to model absolute equality of opportunity.

Here is the sociological and mathematical framework executing under the hood for this specific scenario:

- Wealth Inheritance Abolished: A 99% estate tax (tax_inheritance = 0.99) effectively zeroes out intergenerational economic transfers.

- Educational Meritocracy: Parent-to-child human capital transmission is completely severed (omega_h = 0.0), replaced by a massive universal public endowment (h_base = 8.0) and high stochastic variance (h_variance = 5.0) to simulate individual talent and luck.

- Spatial and Social Integration: Spatial friction (gamma_distance) and homophily (lambda_2) are both set to 0.0. This ensures social networks form entirely randomly across class and geographic lines, preventing elite closure.

- Maximum Redistribution: A 60% income tax feeds the Universal Basic Income pool, and housing market density elasticity (kappa = 0.01) is near zero, effectively neutralizing the bid-rent eviction mechanisms that trapped the poorest agents in previous runs.

Mathematically, this combination of parameters should completely dissolve the main diagonal of the transition matrix. The user should observe the probabilities in every cell stabilizing near 0.20 (20%). In this artificial environment, a descendant's final class position becomes statistically independent of their parent's starting wealth.

## Installation and Execution

To execute the simulation, ensure your Python environment has the necessary dependencies installed. The model requires Mesa version 3.0 or higher.

### 1. Install Dependencies
Execute the following command in your terminal to install the required libraries:

```bash
pip install mesa networkx numpy pandas seaborn matplotlib


### 2. Run the Simulation
Navigate to the application directory and execute the main script:

```bash
python main.py

The command-line interface will prompt you to select a scenario. The application will compute the agent interactions across five generations and render a heatmap visualizing the intergenerational transition probabilities.