# Macro-Sociological Stratification and Mobility Simulator

This repository contains an agent-based model (ABM) developed in Python using the Mesa framework. Using more than 20 parameters found in the literature, it integrates human, social, and economic capital within a spatially explicit urban environment to explore policy-driven redistribution, spatial segregation, and intergenerational mobility. Through defined micro-level structural constraints, the simulation demonstrates how social stratification emerges and inequality reproduces across multiple demographic epochs with various scenarios including real-life examples of the US, Turkey and Finland.

## Theoretical Framework

The simulation architecture operationalizes several foundational sociological and economic theories to generate realistic macroeconomic dynamics.

1. **Multidimensional Capital:** Grounded in Pierre Bourdieu's theory of capital, agents accumulate three distinct resources. Economic capital represents liquid wealth. Human capital represents education and institutional credentials. Social capital represents the economic utility of an agent's network ties.
2. **Spatial Sorting and Bid-Rent Theory:** Agents exist within an urban grid featuring varying institutional quality. The housing market logic forces agents to evaluate their budget constraint against dynamic spatial costs. High-demand, high-quality nodes increase in price, systematically evicting low-capital agents to peripheral zones. 
3. **Network Homophily:** Social networks form based on a gravity model. Agents calculate the probability of forming a tie using spatial distance and class distance. This mechanism simulates elite closure and resource hoarding.
4. **Intergenerational Mobility:** The simulation tracks the transition of capital across biological generations. Offspring inherit a percentage of their parent's economic wealth, a baseline of their human capital (subject to stochastic variance), and fragments of their social network. The model generates transition matrices to quantify social reproduction, comparable to empirical methodologies used in contemporary urban studies and economics.

## Key Components

- **Agents (`app_gui.py` / `agents.py`)** define life-cycle behaviors: social capital summation, Cobb-Douglas income generation, human capital investment, dynamic cost-of-living housing choices, and inheritance with stochastic human-capital shocks.
- **Environment (`app_gui.py` / `model.py`)** maintains the `MultiGrid`, a KD-tree-guided social graph, dynamic rent (`get_dynamic_spatial_cost`), welfare redistribution (`distribute_welfare`), and scenario-specific calibrations for `.gamma_distance`, `.lambda_2`, `.tax_rate`, etc.
- **Metrics (`metrics.py`)** expose Gini, Moran’s I, and mean human capital helpers used by the data collectors.
- **Analysis (`analysis.py`)** captures terminal wealth after each generation, bins parents and children into quintiles, and reports a transition-probability matrix.
- **Entry points** - `python app_gui.py`: launches the ttkbootstrap GUI (default workflow).  
  - `python main.py`: opens a Tkinter prompt, runs five generations with default parameters, and displays a matplotlib heatmap.  

### Income Generation Function

Agents generate income ($Y$) using a modified Cobb-Douglas production function, converting human and social inputs into economic output:

$$Y = A \cdot (C_H)^\alpha \cdot (C_S)^\beta \cdot \epsilon$$

Where $A$ is the base wage, $\alpha$ and $\beta$ are the output elasticities of human and social capital, and $\epsilon$ is a stochastic shock variable representing market variance.

### Network Tie Formation

The probability ($p$) of two agents forming a network tie incorporates both spatial friction and homophily:

$$p = \frac{1}{1 + e^{\gamma_1 d + \gamma_2 |C_i - C_j| - \gamma_0}}$$

Where $d$ is the Euclidean spatial distance, $C$ is the composite class index of the agent, $\gamma_1$ dictates spatial friction, and $\gamma_2$ dictates homophily preference.

## Mathematical Core

The simulation relies on specific mathematical functions to drive agent behavior and capital accumulation. 

- **Income law:** $Y = A \cdot (C_H)^\alpha \cdot (C_S)^\beta \cdot \epsilon$, where $A$ is the base wage, $C_H$ and $C_S$ are human and social capital, and $\epsilon \sim \mathcal{N}(1,0.1)$.
- **Network ties:** $p = \frac{1}{1 + e^{\gamma_1 d + \gamma_2 |C_i - C_j| - \gamma_0}}$ damping spatial distance, while the class-distance weight remains fixed at 0.1.
- **Inheritance:** Agents reset when `age >= max_lifespan`, pay `tax_inheritance`, inherit $h_{\text{base}} + \omega_h \cdot HC + \mathcal{N}(0, \sigma^2)$ human capital, and probabilistically drop old ties based on `rho_net`.

## Macroeconomic Scenarios

The interface will prompt you to select a scenario. The application will compute the agent interactions across generations and render a heatmap visualizing the intergenerational transition probabilities.

| Scenario | Highlights |
| --- | --- |
| Rigid Imaginary | High human-capital persistence, large spatial friction ($\gamma=0.5$), homophily ($\lambda_2=0.1$), expensive housing ($\kappa=0.2$), low redistribution. |
| United States | Moderate persistence, medium friction ($\gamma=0.2$), average housing elasticity, 25% income tax. |
| Turkey | Strong elite closure, slower public education, 15% inheritance tax, steeper subsistence floor ($c_{\text{min}}=15$). |
| Finland | Low friction ($\gamma=0.05$), generous redistribution (`tax_rate=0.45`, `h_base=5`), minimal homophily. |
| Hyper-Mobile Imaginary | $\omega_h=0$, 99% estate tax, zero homophily and friction, near-zero $\kappa$, 60% income tax. |


## What It Simulates

The `UrbanEnvironment` class functions as the macro-sociological framework within the simulation. It acts as the structural container that enforces physical, economic, and social constraints on individual agents, dictating the friction and opportunities they encounter. Functionally, it simulates the systemic forces of a city or nation-state. By adjusting its hyperparameters, the society's economic model and spatial dynamics are fundamentally altered.

Here is a breakdown of its core components and what they represent sociologically:

1. **Spatial Infrastructure and Institutional Quality:** The environment uses a spatial grid (`mesa.space.MultiGrid`) to represent urban geography. Each coordinate on this grid contains an intrinsic `neighborhood_quality` score assigned during initialization. This array simulates the uneven distribution of public goods, such as municipal infrastructure, safety, and school district quality. An agent's location directly bounds its capacity to develop human capital, operationalizing the geographic lottery of birth.
2. **The Dynamic Housing Market:** The environment actively simulates urban economic geography via the `get_dynamic_spatial_cost` method. It calculates localized living expenses by applying a bid-rent mechanism. The base rent of a neighborhood is multiplied by a density elasticity parameter (`kappa`). As high-quality cells attract more agents, the localized demand drives up the cost of living. This component serves as the primary engine for spatial segregation, systematically pricing out low-capital agents and forcing them into under-resourced peripheries. 
3. **Social Network Architecture:** The environment maintains a continuous social graph using `networkx.Graph` to track relationships. Through the `update_networks` method, the macro-environment dictates how social capital forms by applying a gravity model of interaction. It evaluates the spatial distance and the absolute socioeconomic distance between any two agents. By tuning the spatial friction (`gamma_distance`) and homophily (`lambda_2`) parameters, this component simulates phenomena ranging from highly integrated civic communities to severe elite closure and class isolation.
4. **Macroeconomic Policy and The Welfare State:** The environment manages systemic fiscal policy. It collects the taxes generated by individual agents into a central municipal treasury (`tax_pool`). The `distribute_welfare` method then reallocates this wealth evenly across the population at the end of each temporal cycle. This component allows the simulation to test the efficacy of redistributive mechanisms, ranging from entirely laissez-faire capitalism (where the tax rate is zero) to a robust Nordic welfare state featuring a Universal Basic Income (UBI).
5. **Demographic Turnover and Social Reproduction:** The environment acts as the biological clock for the population. It enforces a maximum generational lifespan (`max_lifespan`), triggering simultaneous inheritance events. During these generational turnovers, the environment imposes macro-level inheritance taxes (`tax_inheritance`) and provides a baseline public education endowment (`h_base`) to the offspring. This component controls the mathematical degree to which parent wealth determines descendant outcomes.
6. **The Analytical Engine:** The environment houses a data collection module. At the end of every computational tick, it executes complex sociological metrics to quantify the systemic state. It extracts the Gini coefficient to measure economic inequality and calculates a localized Moran's I to measure the spatial autocorrelation of wealth (geographic segregation).

## Five Epochs as Default

An epoch is the fundamental chronological unit of social reproduction in the model, translating to one complete demographic generation. When the application states it is "simulating across 5 epochs," it is tracking the accumulation and transfer of capital across five distinct generations of agents.

Here is how an epoch functions mathematically and chronologically within the architecture:

* **The Lifespan (The Ticks):** One epoch consists of a specific number of computational time steps (or "ticks"). In `UrbanEnvironment`, this is defined by the `max_lifespan` parameter, which is set to 50. Therefore, one epoch equals 50 ticks of the simulation loop.
* **Intra-Epoch Mechanics (Accumulation):** During these 50 ticks, agents progress through their life cycle. They work to generate income, pay rent in the spatial housing market, pay taxes, invest in their human capital, and form social network ties.
* **The Epoch Boundary (Inheritance):** When an agent reaches the end of the epoch (`age >= self.model.max_lifespan`), the `execute_inheritance()` method is triggered. The agent is replaced by their descendant. The simulation calculates the estate tax, applies human capital variance, and transfers the remaining wealth and social ties to this new generation.
* **The Analytical Output:** To calculate the final intergenerational mobility matrix, the `analysis.py` module takes a snapshot of the agents' wealth at Epoch 1 (the initial parents) and compares it to their descendants' wealth at the end of Epoch 5. 

## The Sociological Takeaway

Despite substantial UBI distributions, the bottom quintile retention rate remained largely unaffected at 87%. Why did the Nordic welfare model fail to save the poorest agents in this environment?

The simulation mathematically demonstrates Universal Inflation and the Rent Trap. In `UrbanEnvironment`, housing costs are dynamically calculated based on neighborhood quality and density (demand). When every agent receives a UBI payment, the baseline purchasing power of the entire population increases simultaneously. 

Because the spatial grid is constrained, this universal influx of cash simply drives up the localized cost of living. The poorest agents received their welfare checks, but those checks were immediately absorbed by the dynamic housing market. They grew richer in absolute terms, but because quintiles measure relative wealth, they remained firmly in the bottom 20% compared to the rest of society.

This demonstrates why sociologists and urban economists argue that UBI alone cannot fix stratification if it is deployed within an unregulated housing market. If income is subsidized without expanding the housing supply or implementing rent controls, welfare capital simply transfers upward to property owners. To achieve high mobility rates, the macro-environment requires decommodified housing (e.g., public housing immune to dynamic market pricing).

## The Hyper-Mobile Imaginary Scenario

The Hyper-Mobile Imaginary scenario pushes the simulation's parameters to their theoretical limits to model absolute equality of opportunity. Here is the sociological and mathematical framework executing within the simulation for this specific scenario:

* **Wealth Inheritance Abolished:** A 99% estate tax (`tax_inheritance = 0.99`) effectively eliminates intergenerational economic transfers.
* **Educational Meritocracy:** Parent-to-child human capital transmission is completely severed (`omega_h = 0.0`), replaced by a substantial universal public endowment (`h_base = 8.0`) and high stochastic variance (`h_variance = 5.0`) to simulate individual talent and luck.
* **Spatial and Social Integration:** Spatial friction (`gamma_distance`) and homophily (`lambda_2`) are both set to 0.0. This ensures social networks form entirely randomly across class and geographic lines, preventing elite closure.
* **Maximum Redistribution:** A 60% income tax feeds the UBI pool, and housing market density elasticity (`kappa = 0.01`) is near zero, effectively neutralizing the bid-rent eviction mechanisms that trapped the poorest agents in previous runs.

Mathematically, this combination of parameters dissolves the main diagonal of the transition matrix. Probabilities in every cell stabilize near 0.20 (20%). In this artificial environment, a descendant's final class position becomes statistically independent of their parent's starting wealth.

## Running the Simulation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python main.py