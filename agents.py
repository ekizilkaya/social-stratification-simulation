import math
import numpy as np
import random
import mesa

class StratificationAgent(mesa.Agent):
    def __init__(self, model, initial_econ, initial_human, age=0):
        super().__init__(model) 
        self.age = age
        self.econ_capital = initial_econ
        self.human_capital = initial_human
        self.social_capital = 0.0
        self.income = 0.0
        
    def calculate_social_capital(self):
        neighbors = list(self.model.social_network.neighbors(self.unique_id))
        s_cap = 0.0
        for n_id in neighbors:
            neighbor = self.model.agent_dict[n_id] 
            s_cap += (self.model.weight_e * neighbor.econ_capital + 
                      self.model.weight_h * neighbor.human_capital)
        self.social_capital = s_cap

    def generate_income(self):
        epsilon = np.random.normal(1.0, 0.1) 
        h_cap = max(self.human_capital, 0.01)
        s_cap = max(self.social_capital, 0.01)
        self.income = (self.model.base_wage * (h_cap ** self.model.alpha) * (s_cap ** self.model.beta) * epsilon)

    def update_human_capital(self):
        cell_quality = self.model.get_neighborhood_quality(self.pos)
        investment_efficiency = math.log(1 + self.model.eta * max(self.econ_capital, 0))
        growth = self.model.theta * cell_quality * investment_efficiency
        depreciation = self.human_capital * self.model.delta_h
        self.human_capital = self.human_capital - depreciation + growth

    def update_economic_capital(self):
        cost_of_living = self.model.get_dynamic_spatial_cost(self.pos)
        tax = self.calculate_tax(self.income, self.econ_capital)
        
        # Transfer the collected tax to the macro-environment's central treasury
        self.model.tax_pool += tax
        
        self.econ_capital = (self.econ_capital * (1 + self.model.interest_rate) + 
                             self.income - cost_of_living - tax)

    def calculate_tax(self, income, wealth):
        return income * self.model.tax_rate

    def evaluate_housing(self):
        current_cost = self.model.get_dynamic_spatial_cost(self.pos)
        budget = (self.model.rho * self.income) + (self.model.phi * max(0, self.econ_capital - self.model.c_min))
        if current_cost > budget:
            self.search_and_relocate(budget)

    def search_and_relocate(self, budget):
        search_radius = 10 
        prospective_cells = []
        for _ in range(search_radius):
            x = self.random.randrange(self.model.grid.width)
            y = self.random.randrange(self.model.grid.height)
            prospective_cells.append((x, y))
            
        best_cell = None
        max_utility = -float('inf')
        agent_class_index = self.econ_capital + self.human_capital + self.social_capital
        
        for cell in prospective_cells:
            cell_cost = self.model.get_dynamic_spatial_cost(cell)
            if cell_cost <= budget:
                cell_agents = self.model.grid.get_cell_list_contents([cell])
                if len(cell_agents) > 0:
                    neighborhood_class_sum = sum(
                        [a.econ_capital + a.human_capital + a.social_capital for a in cell_agents]
                    )
                    mean_neighborhood_class = neighborhood_class_sum / len(cell_agents)
                else:
                    mean_neighborhood_class = agent_class_index 
                    
                quality = self.model.get_neighborhood_quality(cell)
                homophily_penalty = abs(agent_class_index - mean_neighborhood_class)
                utility = (self.model.lambda_1 * quality) - (self.model.lambda_2 * homophily_penalty)
                
                if utility > max_utility:
                    max_utility = utility
                    best_cell = cell
                    
        if best_cell:
            self.model.grid.move_agent(self, best_cell)

    def age_and_reproduce(self):
        self.age += 1
        if self.age >= self.model.max_lifespan:
            self.execute_inheritance()

    def execute_inheritance(self):
        inherited_econ = max(0, self.econ_capital) * (1 - self.model.tax_inheritance)
        
        # Human capital variance is now driven by the macroeconomic scenario
        variance = np.random.normal(0, self.model.h_variance)
        inherited_human = self.model.h_base + (self.model.omega_h * self.human_capital) + variance
        
        self.age = 0
        self.econ_capital = inherited_econ
        self.human_capital = max(1.0, inherited_human) 
        self.income = 0.0
        
        neighbors = list(self.model.social_network.neighbors(self.unique_id))
        for neighbor_id in neighbors:
            if self.random.random() > self.model.rho_net:
                self.model.social_network.remove_edge(self.unique_id, neighbor_id)

    def step(self):
        self.calculate_social_capital()
        self.generate_income()
        self.update_human_capital()
        self.update_economic_capital()
        self.evaluate_housing()
        self.age_and_reproduce()