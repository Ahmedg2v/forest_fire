import numpy as np
from Functions.model_functionality import new_grid
from Functions.visualisation_analysis import animate
# Test fire spread and periodic boundary conditions
# test parameters
x_test = np.arange(0, 11, 1)    # x coordinates
y_test = np.arange(0, 11, 1)    # y coordinates

p_test = 0     # probability of a tree growing in a given cell
f_test = 0     # probability of lightening striking a given cell

# create the grid with all the cells being empty
cells_arr_test = np.ones((len(x_test), len(y_test)), dtype=int)
cells_arr_test[2,2]=2

N_test = 12                             # number of time steps
plot_data_test = []                     # create list to save the grid values at each step
plot_data_test.append(cells_arr_test.copy())  # save the initial state of the grid

for q in range(N_test):
    cells_arr_test = np.array(new_grid(cells_arr_test,p_test,f_test))  # update the grid each time step
    plot_data_test.append(cells_arr_test.copy())             # save the new state
    
animate(plot_data_test, range(N_test), 0.5)