import numpy as np
from Functions.model_functionality import new_grid
from Functions.visualisation_analysis import animate, relative_graph, graph_areas_fit, graph_circumference_fit, grid_n_cluster_map, mean_density

# Simulation & Analysis

## Model parameters
x_c = np.arange(0, 301, 1)    # x coordinates
y_c = np.arange(0, 301, 1)    # y coordinates

p = 0.07       # probability of a tree growing in a given cell
f = 0.000007   # probability of lightening striking a given cell

# Running the simulation

# create the grid with all the cells being empty
cells_arr = np.zeros((len(x_c), len(y_c)), dtype=int)

N = 1000                             # number of time steps
plot_data = []                       # create list to save the grid values at each step
plot_data.append(cells_arr.copy())   # save the initial state of the grid

for q in range(N):
    cells_arr = np.array(new_grid(cells_arr,p,f))  # update the grid each time step
    plot_data.append(cells_arr.copy())             # save the new state

# Equilibrium Check
relative_graph(plot_data,N, cells_arr)

# Critical Behavior Check
graph_areas_fit(plot_data, 2, np.arange(401, 501, 1))
graph_areas_fit(plot_data, 1, np.arange(401, 501, 1))
animate(plot_data, range(N), 0.00001)

# Critical Regime Analysis
grid_n_cluster_map(plot_data,1,-1)
mean_density(plot_data, 1, np.arange(401, 501, 1), cells_arr)
graph_areas_fit(plot_data, 1, np.arange(401, 501, 1))
graph_circumference_fit(plot_data, 1, -1)