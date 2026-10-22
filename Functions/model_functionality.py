import numpy as np

# Model Rules & Functionality

def grow_tree(p, cells_arr):
    '''Checks for empty cells and returns an array of new trees in them according to the probability p'''
    
    # create an array of binomial values with the same shape as the grid (cells_arr) according to probability p 
    seed = np.random.binomial(1, p, (cells_arr.shape[0],cells_arr.shape[1]))

    # returns a boolean array with the same shape as cells_arr. each element [x,y] is 'True' if cell [x,y] in cells_arr 
    # is empty (0) and the corressponding element in the randomly generated array (seed) is equal to 1 and 'False' otherwise
    return (cells_arr == 0) & (seed==1)


def lightening(f, cells_arr):
    '''Strikes cells with trees in them with lightening with probability f'''
    # works in a similar way to grow_tree
    
    # create an array of binomial values with the same shape as the grid (cells_arr) according to probability f
    strike = np.random.binomial(1, f, (cells_arr.shape[0],cells_arr.shape[1]))
    
    # returns a boolean array with the same shape as cells_arr. each element [x,y] is 'True' if cell [x,y] in cells_arr is
    # a tree (1) and the corressponding element in the randomly generated array (strike) is equal to 1, and 'False' otherwise
    return (cells_arr == 1) & (strike==1)


def fire_spread(cells_arr):
    '''Checks whether any neighboring cells are on fire'''
    
    # create an array which wraps for periodic boundary conditions (flat topology to a donut)
    c_wrap = np.pad(cells_arr, pad_width=1, mode='wrap')
    
    # Find adjacent cells that are on fire (2)
    fire_adjacent = ((c_wrap[2:, 1:-1] == 2) |     # up    (x, y+1)
                     (c_wrap[:-2, 1:-1] == 2)|     # down  (x, y-1)
                     (c_wrap[1:-1, 2:] == 2) |     # left  (x-1, y)
                     (c_wrap[1:-1, :-2] == 2) )    # right (x+1, y)
    
    # returns a boolean array with the same shape as cells_arr. each element [x,y] is 'True' if cell [x,y] in cells_arr 
    # is a tree (1) and any of it's adjacent cells is on fire (2), and 'False' otherwise
    return (cells_arr == 1) & fire_adjacent
    
    
def new_grid(cells_arr, p, f):
    '''Updates the grid according to the pre-defined rules and probabilities'''
    
    # empty new grid
    cells_arr_new = np.zeros_like(cells_arr)
    
    # calling other functions
    g_trees = grow_tree(p,cells_arr)
    
    light = lightening(f,cells_arr)
    
    fire = fire_spread(cells_arr)
    
    # cells currently on fire
    ash = (cells_arr == 2)            
    
    # updating the states of the cells
    cells_arr_new[(cells_arr == 1) & (~fire)] = 1   # cells with trees in them and no neighbours on fire are kept as trees
    cells_arr_new[g_trees] = 1                      # growing trees in empty cells
    cells_arr_new[ash] = 0                          # cells on fire become empty
    cells_arr_new[(fire==1)|(light==1)] = 2         # tree cells with neighbours on fire or hit by lightening are set on fire
    
    # return the updated grid
    return cells_arr_new