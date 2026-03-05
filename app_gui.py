import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import networkx as nx
import mesa
import tkinter as tk
from tkinter import messagebox
import locale
_orig_setlocale = locale.setlocale
locale.setlocale = lambda cat, loc="": _orig_setlocale(cat, "C") if loc == "" else _orig_setlocale(cat, loc)
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.widgets import ToolTip
locale.setlocale = _orig_setlocale
import threading
from scipy.spatial import KDTree
import os
import markdown
from tkhtmlview import HTMLLabel

# ==========================================
# 1. METRICS MODULE
# ==========================================
def compute_gini(model):
    agent_wealths = [agent.econ_capital for agent in model.agents]
    x = sorted(agent_wealths)
    n = len(x)
    if n == 0 or sum(x) == 0: return 0.0
    x = np.array(x)
    index = np.arange(1, n + 1)
    return ((np.sum((2 * index - n  - 1) * x)) / (n * np.sum(x)))

def compute_spatial_segregation(model):
    grid_wealth = np.zeros((model.grid.width, model.grid.height))
    for agent in model.agents:
        x, y = agent.pos
        grid_wealth[x][y] += agent.econ_capital
    mean_wealth = np.mean(grid_wealth)
    numerator, denominator, weight_sum = 0.0, 0.0, 0.0
    for x in range(model.grid.width):
        for y in range(model.grid.height):
            cell_val = grid_wealth[x][y] - mean_wealth
            denominator += cell_val ** 2
            neighbors = model.grid.get_neighborhood((x, y), moore=True, include_center=False)
            for nx, ny in neighbors:
                neighbor_val = grid_wealth[nx][ny] - mean_wealth
                numerator += cell_val * neighbor_val
                weight_sum += 1
    if denominator == 0 or weight_sum == 0: return 0.0
    n_cells = model.grid.width * model.grid.height
    return (n_cells / weight_sum) * (numerator / denominator)

def compute_mean_human_capital(model):
    return np.mean([agent.human_capital for agent in model.agents])

# ==========================================
# 2. AGENT MODULE
# ==========================================
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
        tax = self.income * self.model.tax_rate
        self.model.tax_pool += tax # Transfer tax to macroeconomic UBI pool
        self.econ_capital = (self.econ_capital * (1 + self.model.interest_rate) + self.income - cost_of_living - tax)

    def evaluate_housing(self):
        current_cost = self.model.get_dynamic_spatial_cost(self.pos)
        budget = (self.model.rho * self.income) + (self.model.phi * max(0, self.econ_capital - self.model.c_min))
        if current_cost > budget:
            self.search_and_relocate(budget)

    def search_and_relocate(self, budget):
        search_radius = 10 
        prospective_cells = [(self.random.randrange(self.model.grid.width), 
                              self.random.randrange(self.model.grid.height)) for _ in range(search_radius)]
        best_cell = None
        max_utility = -float('inf')
        agent_class_index = self.econ_capital + self.human_capital + self.social_capital
        
        for cell in prospective_cells:
            cell_cost = self.model.get_dynamic_spatial_cost(cell)
            if cell_cost <= budget:
                cell_agents = self.model.grid.get_cell_list_contents([cell])
                if len(cell_agents) > 0:
                    mean_neighborhood_class = sum([a.econ_capital + a.human_capital + a.social_capital for a in cell_agents]) / len(cell_agents)
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
            inherited_econ = max(0, self.econ_capital) * (1 - self.model.tax_inheritance)
            variance = np.random.normal(0, self.model.h_variance)
            inherited_human = self.model.h_base + (self.model.omega_h * self.human_capital) + variance
            
            self.age = 0
            self.econ_capital = inherited_econ
            self.human_capital = max(1.0, inherited_human) 
            self.income = 0.0
            
            neighbors = list(self.model.social_network.neighbors(self.unique_id))
            for n_id in neighbors:
                if self.random.random() > self.model.rho_net:
                    self.model.social_network.remove_edge(self.unique_id, n_id)

    def step(self):
        self.calculate_social_capital()
        self.generate_income()
        self.update_human_capital()
        self.update_economic_capital()
        self.evaluate_housing()
        self.age_and_reproduce()

# ==========================================
# 3. ENVIRONMENT MODULE
# ==========================================
class UrbanEnvironment(mesa.Model):
    def __init__(self, num_agents, width, height, scenario="United States", **kwargs):
        super().__init__()
        self.num_agents = int(num_agents)
        self.grid = mesa.space.MultiGrid(int(width), int(height), torus=False)
        self.social_network = nx.Graph()
        self.agent_dict = {} 
        self.tax_pool = 0.0
        
        # Universal Base Hyperparameters
        self.base_wage = 10.0
        self.alpha, self.beta = 0.3, 0.2           
        self.delta_h, self.theta, self.eta = 0.05, 0.1, 0.01          
        self.weight_e, self.weight_h = 0.5, 0.5       
        self.interest_rate, self.mu = 0.02, 15.0            
        self.rho, self.phi = 0.3, 0.05           
        self.lambda_1, self.max_lifespan, self.rho_net = 2.0, 50, 0.7           
        
        self.set_scenario_parameters(scenario)
        for k, v in kwargs.items():
            setattr(self, k, float(v) if not isinstance(v, int) else v)

        self.neighborhood_quality = np.random.uniform(0.5, 1.5, (int(width), int(height)))
        self.initialize_agents()

    def set_scenario_parameters(self, scenario):
        if scenario == "Rigid Imaginary":
            self.omega_h, self.h_variance, self.tax_inheritance = 0.6, 1.0, 0.40
            self.gamma_distance, self.lambda_2, self.kappa = 0.5, 0.1, 0.2
            self.h_base, self.c_min, self.tax_rate = 2.0, 5.0, 0.15
        elif scenario == "United States":
            self.omega_h, self.h_variance, self.tax_inheritance = 0.4, 2.0, 0.40 
            self.gamma_distance, self.lambda_2, self.kappa = 0.2, 0.05, 0.1
            self.h_base, self.c_min, self.tax_rate = 2.0, 5.0, 0.25        
        elif scenario == "Turkey":
            self.omega_h, self.h_variance, self.tax_inheritance = 0.50, 2.0, 0.15 
            self.gamma_distance, self.lambda_2, self.kappa = 0.25, 0.08, 0.10
            self.h_base, self.c_min, self.tax_rate = 3.0, 15.0, 0.20        
        elif scenario == "Finland":
            self.omega_h, self.h_variance, self.tax_inheritance = 0.15, 3.0, 0.60
            self.gamma_distance, self.lambda_2, self.kappa = 0.05, 0.01, 0.05
            self.h_base, self.c_min, self.tax_rate = 5.0, 5.0, 0.45        
        elif scenario == "Hyper-Mobile Imaginary":
            self.omega_h, self.h_variance, self.tax_inheritance = 0.0, 5.0, 0.99 
            self.gamma_distance, self.lambda_2, self.kappa = 0.0, 0.0, 0.01
            self.h_base, self.c_min, self.tax_rate = 8.0, 5.0, 0.60        

    def initialize_agents(self):
        for i in range(self.num_agents):
            init_econ = max(np.random.normal(100, 30), 10)
            init_human = max(np.random.normal(10, 3), 1)
            agent = StratificationAgent(self, init_econ, init_human, age=0)
            self.agent_dict[agent.unique_id] = agent
            self.social_network.add_node(agent.unique_id)
            self.grid.place_agent(agent, (self.random.randrange(self.grid.width), self.random.randrange(self.grid.height)))

    def get_neighborhood_quality(self, pos):
        return self.neighborhood_quality[pos[0]][pos[1]]

    def get_dynamic_spatial_cost(self, pos):
        density = len(self.grid.get_cell_list_contents([pos]))
        return self.mu * self.get_neighborhood_quality(pos) * (1 + (self.kappa * density))

    def update_networks(self):
        nodes = list(self.agent_dict.keys())
        if not nodes: return
        
        # Calculate effective spatial radius for tie formation to prune brute-force checks
        # Using a conservative threshold where probability fundamentally approaches 0
        if self.gamma_distance > 0.001:
            effective_radius = 10.0 / self.gamma_distance
        else:
            effective_radius = max(self.grid.width, self.grid.height) * 2
            
        coords = np.array([self.agent_dict[n].pos for n in nodes])
        tree = KDTree(coords)
        close_pairs = tree.query_pairs(r=effective_radius)
        
        pairs_to_check = set()
        for i, j in close_pairs:
            pairs_to_check.add((nodes[i], nodes[j]))
            
        for u, v in list(self.social_network.edges):
            if (u, v) not in pairs_to_check and (v, u) not in pairs_to_check:
                pairs_to_check.add((u, v))
                
        for n_i, n_j in pairs_to_check:
            agent_i = self.agent_dict[n_i]
            agent_j = self.agent_dict[n_j]
            dist = math.dist(agent_i.pos, agent_j.pos)
            cap_diff = abs((agent_i.econ_capital + agent_i.human_capital) - (agent_j.econ_capital + agent_j.human_capital))
            logit = - (self.gamma_distance * dist) - (0.1 * cap_diff) + 2.0
            prob = 0.0 if logit < -700 else 1.0 / (1.0 + math.exp(-logit))
            
            if self.random.random() < prob: 
                self.social_network.add_edge(n_i, n_j)
            elif self.social_network.has_edge(n_i, n_j): 
                self.social_network.remove_edge(n_i, n_j)

    def distribute_welfare(self):
        if self.tax_pool > 0 and self.num_agents > 0:
            ubi_per_agent = self.tax_pool / self.num_agents
            for agent in self.agents: agent.econ_capital += ubi_per_agent
            self.tax_pool = 0.0

    def step(self):
        self.update_networks()
        self.agents.shuffle_do("step") 
        self.distribute_welfare()

# ==========================================
# 4. ANALYSIS MODULE
# ==========================================
def run_simulation_logic(scenario, generations=5, num_agents=500, width=20, height=20,
                         progress_callback=None, cancel_event=None, **kwargs):
    model = UrbanEnvironment(num_agents=num_agents, width=width, height=height, scenario=scenario, **kwargs)
    terminal_wealth_records = []
    total_ticks = int(generations) * int(model.max_lifespan)
    
    for tick in range(total_ticks):
        if cancel_event and cancel_event.is_set():
            return None
        model.step()
        if progress_callback:
            progress_callback(tick + 1, total_ticks)
        if (tick + 1) % model.max_lifespan == 0:
            wealth_data = {
                'Agent_ID': [a.unique_id for a in model.agents],
                'Generation': (tick + 1) // model.max_lifespan,
                'Economic_Capital': [a.econ_capital for a in model.agents]
            }
            terminal_wealth_records.append(pd.DataFrame(wealth_data))

    data = pd.concat(terminal_wealth_records, ignore_index=True)
    parents = data[data['Generation'] == 1].copy()
    children = data[data['Generation'] == generations].copy()
    
    parents['Parent_Quintile'] = pd.qcut(parents['Economic_Capital'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    children['Child_Quintile'] = pd.qcut(children['Economic_Capital'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    
    mobility_df = pd.merge(parents[['Agent_ID', 'Parent_Quintile']], children[['Agent_ID', 'Child_Quintile']], on='Agent_ID')
    transition_counts = pd.crosstab(mobility_df['Parent_Quintile'], mobility_df['Child_Quintile'])
    return transition_counts.div(transition_counts.sum(axis=1), axis=0)

# ==========================================
# 5. GUI CONSTANTS
# ==========================================

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

# ==========================================
# 6. GUI & EXECUTION MODULE
# ==========================================

class SimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Social Stratification Simulator")

        self.cancel_event = threading.Event()
        self._last_pct = -1
        self.param_vars = {}

        self.scenarios = list(SCENARIO_DESCRIPTIONS.keys())

        self.default_params = {
            "num_agents": 500, "width": 20, "height": 20,
            "base_wage": 10.0, "alpha": 0.3, "beta": 0.2,
            "delta_h": 0.05, "theta": 0.1, "eta": 0.01,
            "weight_e": 0.5, "weight_h": 0.5, "interest_rate": 0.02,
            "mu": 15.0, "rho": 0.3, "phi": 0.05,
            "lambda_1": 2.0, "max_lifespan": 50, "rho_net": 0.7,
            "generations": 5
        }

        self.params_config = {
            "General": ["num_agents", "width", "height"],
            "Macroeconomic": ["tax_rate", "tax_inheritance", "c_min", "base_wage", "interest_rate"],
            "Education": ["h_base", "omega_h", "h_variance", "delta_h", "theta", "eta"],
            "Urban & Housing": ["mu", "kappa", "rho", "phi", "lambda_1"],
            "Social Network": ["gamma_distance", "lambda_2", "alpha", "beta", "weight_e", "weight_h", "rho_net"],
            "Temporal": ["max_lifespan", "generations"]
        }

        self._build_ui()
        self.on_scenario_change()

    # ── UI Construction ─────────────────────────────────────

    def _build_ui(self):
        # ── Header ──
        header = ttk.Frame(self.root, bootstyle="primary", padding=(20, 14))
        header.pack(fill=X)

        title_area = ttk.Frame(header, bootstyle="primary")
        title_area.pack(side=LEFT)
        ttk.Label(title_area, text="Social Stratification Simulator",
                  font=("Segoe UI", 17, "bold"),
                  bootstyle="inverse-primary").pack(anchor=W)
        ttk.Label(title_area,
                  text="Agent-based intergenerational mobility & spatial segregation model",
                  font=("Segoe UI", 9),
                  bootstyle="inverse-primary").pack(anchor=W, pady=(2, 0))

        header_btns = ttk.Frame(header, bootstyle="primary")
        header_btns.pack(side=RIGHT)
        ttk.Button(header_btns, text="Parameter Guide",
                   command=self.show_help,
                   bootstyle="light").pack(side=LEFT, padx=4)
        ttk.Button(header_btns, text="Read Me",
                   command=self.show_readme,
                   bootstyle="light").pack(side=LEFT, padx=4)

        # ── Scenario Selection ──
        scenario_frame = ttk.LabelFrame(self.root, text="  Scenario  ")
        scenario_frame.pack(fill=X, padx=16, pady=(12, 0))

        scenario_inner = ttk.Frame(scenario_frame, padding=(14, 10))
        scenario_inner.pack(fill=X)

        row = ttk.Frame(scenario_inner)
        row.pack(fill=X)

        self.scenario_var = tk.StringVar()
        self.dropdown = ttk.Combobox(row, textvariable=self.scenario_var,
                                     values=self.scenarios, state="readonly",
                                     width=28, font=("Segoe UI", 10))
        self.dropdown.current(1)
        self.dropdown.pack(side=LEFT, padx=(0, 10))
        self.dropdown.bind("<<ComboboxSelected>>", self.on_scenario_change)

        ttk.Button(row, text="Reset to Defaults", command=self.reset_defaults,
                   bootstyle="secondary-outline").pack(side=LEFT)

        self.scenario_desc_label = ttk.Label(scenario_inner, text="",
                                             wraplength=860,
                                             font=("Segoe UI", 9, "italic"),
                                             foreground="gray")
        self.scenario_desc_label.pack(fill=X, pady=(8, 2))

        # ── Main Content (Configuration / Results) ──
        self.main_notebook = ttk.Notebook(self.root, bootstyle="primary")
        self.main_notebook.pack(expand=True, fill=BOTH, padx=16, pady=10)

        # Configuration Tab
        config_outer = ttk.Frame(self.main_notebook, padding=6)
        self.main_notebook.add(config_outer, text="  Configuration  ")

        self.param_notebook = ttk.Notebook(config_outer)
        self.param_notebook.pack(expand=True, fill=BOTH)

        for cat, params in self.params_config.items():
            tab = ttk.Frame(self.param_notebook, padding=(12, 10))
            self.param_notebook.add(tab, text=f"  {cat}  ")

            for i, p in enumerate(params):
                display_name, description = PARAM_INFO.get(p, (p, ""))

                lbl = ttk.Label(tab, text=display_name,
                                font=("Segoe UI", 10), width=26, anchor=W)
                lbl.grid(row=i, column=0, padx=(4, 10), pady=7, sticky=W)

                var = tk.StringVar()
                self.param_vars[p] = var
                entry = ttk.Entry(tab, textvariable=var, width=12,
                                  font=("Segoe UI", 10))
                entry.grid(row=i, column=1, padx=(0, 12), pady=7, sticky=W)

                desc_lbl = ttk.Label(tab, text=description,
                                     font=("Segoe UI", 8, "italic"),
                                     foreground="gray")
                desc_lbl.grid(row=i, column=2, padx=(0, 4), pady=7, sticky=W)

                ToolTip(entry, text=f"{p}: {description}", bootstyle="info-inverse")
                ToolTip(lbl, text=f"Parameter key: {p}", bootstyle="info-inverse")

            tab.columnconfigure(2, weight=1)

        # Results Tab
        self.results_frame = ttk.Frame(self.main_notebook, padding=6)
        self.main_notebook.add(self.results_frame, text="  Results  ")

        self._show_results_placeholder(
            "Run a simulation to see results here.\n\n"
            "Select a scenario, adjust parameters, then click 'Run Simulation'.")

        # ── Action Bar ──
        action_frame = ttk.Frame(self.root, padding=(16, 0, 16, 4))
        action_frame.pack(fill=X)

        ttk.Separator(action_frame).pack(fill=X, pady=(0, 10))

        btn_row = ttk.Frame(action_frame)
        btn_row.pack(fill=X, pady=(0, 6))

        self.run_button = ttk.Button(
            btn_row, text="Run Simulation", command=self.run_simulation,
            bootstyle="success", padding=(16, 8))
        self.run_button.pack(side=LEFT)

        self.cancel_button = ttk.Button(
            btn_row, text="Cancel", command=self.cancel_simulation,
            bootstyle="danger-outline", padding=(12, 8))
        self.cancel_button.pack(side=LEFT, padx=(8, 0))
        self.cancel_button.config(state="disabled")

        progress_row = ttk.Frame(action_frame)
        progress_row.pack(fill=X, pady=(0, 6))

        self.progress = ttk.Progressbar(
            progress_row, bootstyle="success-striped", orient=HORIZONTAL,
            length=400, mode="determinate", maximum=100)
        self.progress.pack(side=LEFT, fill=X, expand=True, padx=(0, 10))

        self.progress_label = ttk.Label(progress_row, text="", width=6,
                                        font=("Segoe UI", 9, "bold"))
        self.progress_label.pack(side=RIGHT)

        # ── Status Bar ──
        status_bar = ttk.Frame(self.root, padding=(16, 2, 16, 8))
        status_bar.pack(fill=X)

        self.status_label = ttk.Label(status_bar, text="Ready",
                                      font=("Segoe UI", 9))
        self.status_label.pack(side=LEFT)

    def _show_results_placeholder(self, message):
        for w in self.results_frame.winfo_children():
            w.destroy()
        ttk.Label(self.results_frame, text=f"\n\n\n\n{message}",
                  font=("Segoe UI", 12), foreground="gray",
                  anchor=CENTER, justify=CENTER).pack(expand=True, fill=BOTH)

    # ── Event Handlers ──────────────────────────────────────

    def on_scenario_change(self, event=None):
        scenario = self.scenario_var.get()
        self.scenario_desc_label.config(
            text=SCENARIO_DESCRIPTIONS.get(scenario, ""))
        temp_env = UrbanEnvironment(num_agents=500, width=20, height=20,
                                    scenario=scenario)
        for p, var in self.param_vars.items():
            if hasattr(temp_env, p):
                var.set(str(getattr(temp_env, p)))
            elif p in self.default_params:
                var.set(str(self.default_params[p]))

    def reset_defaults(self):
        self.on_scenario_change()

    def show_help(self):
        win = tk.Toplevel(self.root)
        win.title("Parameter Guide")
        win.geometry("700x560")
        win.transient(self.root)

        container = ttk.Frame(win, padding=12)
        container.pack(expand=True, fill=BOTH)

        text = tk.Text(container, wrap=tk.WORD, font=("Segoe UI", 10),
                       padx=16, pady=12, spacing1=1, spacing3=2,
                       relief="flat", borderwidth=0)
        scrollbar = ttk.Scrollbar(container, orient=VERTICAL,
                                  command=text.yview, bootstyle="round")
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=RIGHT, fill=Y)
        text.pack(expand=True, fill=BOTH)

        text.tag_configure("title", font=("Segoe UI", 14, "bold"),
                           spacing1=6, spacing3=10)
        text.tag_configure("cat", font=("Segoe UI", 12, "bold"),
                           spacing1=14, spacing3=6, foreground="#2c3e50")
        text.tag_configure("pname", font=("Segoe UI", 10, "bold"),
                           foreground="#2980b9")
        text.tag_configure("pkey", font=("Consolas", 9), foreground="#7f8c8d")
        text.tag_configure("pdesc", font=("Segoe UI", 10), foreground="#34495e")

        text.insert(tk.END, "Parameter Reference Guide\n\n", "title")

        for cat, params in self.params_config.items():
            text.insert(tk.END, f"\n{cat}\n", "cat")
            for p in params:
                display_name, description = PARAM_INFO.get(p, (p, ""))
                text.insert(tk.END, f"  {display_name}  ", "pname")
                text.insert(tk.END, f"  ({p})\n", "pkey")
                text.insert(tk.END, f"    {description}\n\n", "pdesc")

        text.config(state=tk.DISABLED)

    def show_readme(self):
        win = tk.Toplevel(self.root)
        win.title("Read Me — Social Stratification Simulator")
        win.geometry("700x560")
        win.transient(self.root)

        try:
            with open("readme.txt", "r", encoding="utf-8") as f:
                md_text = f.read()
        except Exception as e:
            md_text = f"**Error loading readme.txt**: {e}"

        html_content = markdown.markdown(md_text)
        html_label = HTMLLabel(
            win,
            html=f"<div style='font-family: Segoe UI, Arial; "
                 f"padding: 14px; line-height: 1.5;'>{html_content}</div>")
        html_label.pack(expand=True, fill=BOTH, padx=10, pady=10)

    # ── Simulation Execution ────────────────────────────────

    def run_simulation(self):
        scenario = self.scenario_var.get()
        kwargs = {}
        for p, var in self.param_vars.items():
            val = var.get().strip()
            try:
                kwargs[p] = float(val) if '.' in val else int(val)
            except ValueError:
                label = PARAM_INFO.get(p, (p, ""))[0]
                messagebox.showerror("Invalid Parameter",
                                     f"Invalid value for '{label}' ({p}): {val}")
                return

        self.cancel_event.clear()
        self._last_pct = -1
        self.run_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        self.dropdown.config(state="disabled")
        self.progress["value"] = 0
        self.progress_label.config(text="0%")
        self.status_label.config(text=f"Simulating '{scenario}'...",
                                 foreground="#2980b9")

        self._show_results_placeholder(
            f"Simulation in progress...\n\nScenario: {scenario}")
        self.main_notebook.select(1)

        threading.Thread(target=self._run_thread,
                         args=(scenario, kwargs), daemon=True).start()

    def cancel_simulation(self):
        self.cancel_event.set()
        self.status_label.config(text="Cancelling...", foreground="orange")

    def _progress_callback(self, current, total):
        pct = int((current / total) * 100)
        if pct != self._last_pct:
            self._last_pct = pct
            self.root.after(0, self._set_progress, pct)

    def _set_progress(self, pct):
        self.progress["value"] = pct
        self.progress_label.config(text=f"{pct}%")

    def _run_thread(self, scenario, kwargs):
        try:
            gen = int(kwargs.pop("generations", 5))
            num_a = int(kwargs.pop("num_agents", 500))
            w = int(kwargs.pop("width", 20))
            h = int(kwargs.pop("height", 20))

            matrix = run_simulation_logic(
                scenario=scenario, generations=gen,
                num_agents=num_a, width=w, height=h,
                progress_callback=self._progress_callback,
                cancel_event=self.cancel_event,
                **kwargs)

            if matrix is None:
                self.root.after(0, self._on_cancelled)
            else:
                self.root.after(0, self._on_complete, matrix, scenario)
        except Exception as e:
            self.root.after(0, self._on_error, str(e))

    def _restore_controls(self):
        self.run_button.config(state="normal")
        self.cancel_button.config(state="disabled")
        self.dropdown.config(state="readonly")

    def _on_cancelled(self):
        self.progress["value"] = 0
        self.progress_label.config(text="")
        self._restore_controls()
        self.status_label.config(text="Simulation cancelled.", foreground="orange")
        self._show_results_placeholder(
            "Simulation was cancelled.\n\nAdjust parameters and try again.")

    def _on_complete(self, matrix, scenario):
        self.progress["value"] = 100
        self.progress_label.config(text="100%")
        self._restore_controls()
        self.status_label.config(text="Simulation complete!", foreground="green")

        # Clear results area
        for w in self.results_frame.winfo_children():
            w.destroy()

        # Title
        ttk.Label(self.results_frame,
                  text=f"Intergenerational Mobility \u2014 {scenario}",
                  font=("Segoe UI", 14, "bold")).pack(pady=(6, 4))

        # Heatmap
        fig = Figure(figsize=(7, 4.8), dpi=100)
        fig.patch.set_facecolor("#fafafa")
        ax = fig.add_subplot(111)
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax,
                    cbar=True, linewidths=0.8, linecolor="white",
                    annot_kws={"size": 12, "weight": "bold"},
                    cbar_kws={"shrink": 0.85, "label": "Probability"})
        ax.set_title("Transition Probability Matrix",
                     fontsize=12, pad=10, fontweight="bold")
        ax.set_ylabel("Parent Quintile (Gen 1)", fontsize=10)
        ax.set_xlabel("Child Quintile (Gen N)", fontsize=10)
        ax.set_facecolor("#fafafa")
        fig.tight_layout(pad=1.5)

        canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=6, pady=(0, 4))

        # Matplotlib toolbar for zoom/pan/save
        toolbar_frame = ttk.Frame(self.results_frame)
        toolbar_frame.pack(fill=X, padx=6)
        NavigationToolbar2Tk(canvas, toolbar_frame)

        # Summary statistics
        if hasattr(matrix, "values"):
            diag = np.diag(matrix.values)
            avg_persistence = np.mean(diag)
            q1_ret = diag[0] if len(diag) > 0 else 0
            q5_ret = diag[-1] if len(diag) > 0 else 0

            stats = ttk.LabelFrame(self.results_frame,
                                   text="  Summary Statistics  ")
            stats.pack(fill=X, padx=6, pady=(6, 4))

            stats_inner = ttk.Frame(stats, padding=(12, 8))
            stats_inner.pack(fill=X)

            stats_row = ttk.Frame(stats_inner)
            stats_row.pack(fill=X)

            for label, value, style in [
                ("Avg. Diagonal Persistence", f"{avg_persistence:.1%}", "info"),
                ("Bottom Quintile Retention", f"{q1_ret:.1%}", "warning"),
                ("Top Quintile Retention", f"{q5_ret:.1%}", "success"),
            ]:
                cell = ttk.Frame(stats_row)
                cell.pack(side=LEFT, expand=True, fill=X, padx=4)
                ttk.Label(cell, text=label, font=("Segoe UI", 8),
                          foreground="gray").pack(anchor=CENTER)
                ttk.Label(cell, text=value,
                          font=("Segoe UI", 14, "bold"),
                          bootstyle=style).pack(anchor=CENTER)

        self.main_notebook.select(1)

    def _on_error(self, err_msg):
        self.progress["value"] = 0
        self.progress_label.config(text="")
        self._restore_controls()
        self.status_label.config(text="Simulation error.", foreground="red")
        messagebox.showerror("Simulation Error", err_msg)
        self._show_results_placeholder(
            f"Simulation encountered an error.\n\n{err_msg}")


if __name__ == "__main__":
    root = ttk.Window(themename="cosmo", title="Social Stratification Simulator",
                      size=(960, 750), resizable=(True, True))
    app = SimulationApp(root)

    root.place_window_center()
    root.lift()
    root.attributes("-topmost", True)
    root.after_idle(root.attributes, "-topmost", False)
    root.focus_force()

    root.mainloop()
