import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from model import UrbanEnvironment

def run_mobility_simulation(scenario, generations=4, num_agents=500, width=20, height=20, **kwargs):
    model = UrbanEnvironment(num_agents=num_agents, width=width, height=height, scenario=scenario, **kwargs)
    terminal_wealth_records = []
    
    total_ticks = generations * model.max_lifespan
    
    for tick in range(total_ticks):
        model.step()
        
        if (tick + 1) % model.max_lifespan == 0:
            generation_index = (tick + 1) // model.max_lifespan
            
            wealth_data = {
                'Agent_ID': [agent.unique_id for agent in model.agents],
                'Generation': generation_index,
                'Economic_Capital': [agent.econ_capital for agent in model.agents]
            }
            df_gen = pd.DataFrame(wealth_data)
            terminal_wealth_records.append(df_gen)

    longitudinal_data = pd.concat(terminal_wealth_records, ignore_index=True)
    return compute_transition_matrix(longitudinal_data, generations)

def compute_transition_matrix(data, total_generations):
    parents = data[data['Generation'] == 1].copy()
    children = data[data['Generation'] == total_generations].copy()
    
    parents['Parent_Quintile'] = pd.qcut(parents['Economic_Capital'], q=5, labels=[1, 2, 3, 4, 5])
    children['Child_Quintile'] = pd.qcut(children['Economic_Capital'], q=5, labels=[1, 2, 3, 4, 5])
    
    mobility_df = pd.merge(
        parents[['Agent_ID', 'Parent_Quintile']], 
        children[['Agent_ID', 'Child_Quintile']], 
        on='Agent_ID'
    )
    
    transition_counts = pd.crosstab(
        mobility_df['Parent_Quintile'], 
        mobility_df['Child_Quintile']
    )
    
    transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0)
    return transition_matrix

def visualize_mobility_matrix(matrix, scenario_name):
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(
        matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="YlGnBu", 
        vmin=0.0, 
        vmax=1.0,
        cbar_kws={'label': 'Transition Probability'}
    )
    
    plt.title(f'Intergenerational Mobility Matrix\nScenario: {scenario_name}')
    plt.xlabel('Descendant Wealth Quintile')
    plt.ylabel('Parent Wealth Quintile')
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()