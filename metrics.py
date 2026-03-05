import numpy as np
import mesa

def compute_gini(model):
    agent_wealths = [agent.econ_capital for agent in model.agents]
    x = sorted(agent_wealths)
    n = len(x)
    
    if n == 0 or sum(x) == 0:
        return 0.0
        
    x = np.array(x)
    index = np.arange(1, n + 1)
    return ((np.sum((2 * index - n  - 1) * x)) / (n * np.sum(x)))

def compute_spatial_segregation(model):
    grid_wealth = np.zeros((model.grid.width, model.grid.height))
    
    for agent in model.agents:
        x, y = agent.pos
        grid_wealth[x][y] += agent.econ_capital
        
    mean_wealth = np.mean(grid_wealth)
    
    numerator = 0.0
    denominator = 0.0
    weight_sum = 0.0
    
    for x in range(model.grid.width):
        for y in range(model.grid.height):
            cell_val = grid_wealth[x][y] - mean_wealth
            denominator += cell_val ** 2
            
            neighbors = model.grid.get_neighborhood((x, y), moore=True, include_center=False)
            for nx, ny in neighbors:
                neighbor_val = grid_wealth[nx][ny] - mean_wealth
                numerator += cell_val * neighbor_val
                weight_sum += 1

    if denominator == 0 or weight_sum == 0:
        return 0.0
        
    n_cells = model.grid.width * model.grid.height
    morans_i = (n_cells / weight_sum) * (numerator / denominator)
    return morans_i

def compute_mean_human_capital(model):
    human_capitals = [agent.human_capital for agent in model.agents]
    return np.mean(human_capitals)