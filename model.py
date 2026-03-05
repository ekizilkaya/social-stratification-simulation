import numpy as np
import networkx as nx
import math
import mesa

from agents import StratificationAgent
from metrics import compute_gini, compute_spatial_segregation, compute_mean_human_capital

class UrbanEnvironment(mesa.Model):
    def __init__(self, num_agents, width, height, scenario="United States"):
        super().__init__()
        self.num_agents = num_agents
        self.grid = mesa.space.MultiGrid(width, height, torus=False)
        self.social_network = nx.Graph()
        self.agent_dict = {} 
        
        # Universal Base Hyperparameters
        self.base_wage = 10.0
        self.alpha = 0.3          
        self.beta = 0.2           
        self.delta_h = 0.05       
        self.theta = 0.1          
        self.eta = 0.01           
        self.weight_e = 0.5       
        self.weight_h = 0.5       
        self.interest_rate = 0.02
        self.mu = 15.0            
        self.rho = 0.3            
        self.phi = 0.05           
        self.c_min = 5.0          
        self.lambda_1 = 2.0       
        self.max_lifespan = 50       
        self.rho_net = 0.7   
        self.tax_pool = 0.0       # Initialize the central tax treasury
        self.base_wage = 10.0        
        
        # Inject scenario-specific parameters
        self.set_scenario_parameters(scenario)
        
        self.neighborhood_quality = np.random.uniform(0.5, 1.5, (width, height))
        self.initialize_agents()
        
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Gini": compute_gini,
                "Spatial_Segregation": compute_spatial_segregation,
                "Mean_Human_Capital": compute_mean_human_capital
            },
            agent_reporters={
                "Economic_Capital": "econ_capital",
                "Human_Capital": "human_capital",
                "Social_Capital": "social_capital"
            }
        )

    def set_scenario_parameters(self, scenario):
        """Calibrates structural constraints to approximate specific empirical realities."""
        if scenario == "Rigid Imaginary":
            self.omega_h = 0.6          # High parent-to-child education transmission 
            self.h_variance = 1.0       # Low variance/luck
            self.tax_inheritance = 0.40 # Moderate estate tax
            self.gamma_distance = 0.5   # High spatial network friction
            self.lambda_2 = 0.1         # High homophily preference
            self.kappa = 0.2            # Highly rigid, expensive housing
            self.h_base = 2.0           # Low public school baseline
            self.tax_rate = 0.15        # Low redistributive tax
            
        elif scenario == "United States":
            self.omega_h = 0.4          # Moderate-high transmission
            self.h_variance = 2.0       # Moderate luck/variance
            self.tax_inheritance = 0.40 
            self.gamma_distance = 0.2   # Moderate spatial friction
            self.lambda_2 = 0.05        # Moderate homophily
            self.kappa = 0.1            # Average housing market density elasticity
            self.h_base = 2.0           
            self.tax_rate = 0.25        
            
        elif scenario == "Turkey":
            self.omega_h = 0.50         # Lowered slightly to allow for some educational mobility
            self.h_variance = 2.0       # Increased to introduce more stochastic "luck"
            self.tax_inheritance = 0.15 
            self.gamma_distance = 0.25  
            self.lambda_2 = 0.08        
            self.kappa = 0.10           # Lowered to reduce housing market eviction rates
            self.h_base = 3.0           # Increased base public education floor
            self.c_min = 15.0           # Increased subsistence threshold to prevent total bankruptcy
            self.tax_rate = 0.20
            
        elif scenario == "Finland":
            self.omega_h = 0.15         # Very low transmission (parents matter less)
            self.h_variance = 3.0       # High variance
            self.tax_inheritance = 0.60 # High estate tax
            self.gamma_distance = 0.05  # Highly mixed networks
            self.lambda_2 = 0.01        # Negligible homophily
            self.kappa = 0.05           # Subsidized/flexible housing
            self.h_base = 5.0           # Excellent universal baseline education
            self.tax_rate = 0.45        # High redistributive tax
            
        elif scenario == "Hyper-Mobile Imaginary":
            self.omega_h = 0.0          # Complete equality of opportunity
            self.h_variance = 5.0       # Outcomes driven entirely by variance
            self.tax_inheritance = 0.99 # 99% estate tax
            self.gamma_distance = 0.0   # Distance has zero effect on networks
            self.lambda_2 = 0.0         # Zero homophily
            self.kappa = 0.01           # Perfect housing liquidity
            self.h_base = 8.0           # Universal massive public endowment
            self.tax_rate = 0.60        

    def initialize_agents(self):
        for i in range(self.num_agents):
            init_econ = max(np.random.normal(100, 30), 10)
            init_human = max(np.random.normal(10, 3), 1)
            agent = StratificationAgent(self, init_econ, init_human, age=0)
            self.agent_dict[agent.unique_id] = agent
            self.social_network.add_node(agent.unique_id)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def get_neighborhood_quality(self, pos):
        x, y = pos
        return self.neighborhood_quality[x][y]

    def get_dynamic_spatial_cost(self, pos):
        base_quality = self.get_neighborhood_quality(pos)
        cell_agents = self.grid.get_cell_list_contents([pos])
        density = len(cell_agents)
        return self.mu * base_quality * (1 + (self.kappa * density))

    def update_networks(self):
        nodes = list(self.social_network.nodes)
        for i in range(len(nodes)):
            agent_i = self.agent_dict[nodes[i]] 
            for j in range(i + 1, len(nodes)):
                agent_j = self.agent_dict[nodes[j]]
                
                dist = math.dist(agent_i.pos, agent_j.pos)
                cap_diff = abs((agent_i.econ_capital + agent_i.human_capital) - 
                               (agent_j.econ_capital + agent_j.human_capital))
                
                logit = - (self.gamma_distance * dist) - (0.1 * cap_diff) + 2.0
                
                if logit < -700:
                    prob = 0.0
                else:
                    prob = 1.0 / (1.0 + math.exp(-logit))
                
                if self.random.random() < prob:
                    self.social_network.add_edge(nodes[i], nodes[j])
                else:
                    if self.social_network.has_edge(nodes[i], nodes[j]):
                        self.social_network.remove_edge(nodes[i], nodes[j])
    def distribute_welfare(self):
        """
        Redistributes all collected taxes equally among all agents as a Universal Basic Income.
        Operationalizes the Nordic welfare state model.
        """
        if self.tax_pool > 0 and self.num_agents > 0:
            ubi_per_agent = self.tax_pool / self.num_agents
            
            for agent in self.agents:
                agent.econ_capital += ubi_per_agent
                
            # Reset the treasury to zero for the next temporal tick
            self.tax_pool = 0.0

    def step(self):
        self.update_networks()
        self.agents.shuffle_do("step") 
        self.datacollector.collect(self)