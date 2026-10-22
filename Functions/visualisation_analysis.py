import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw,pause,close
import matplotlib
from scipy import ndimage
from scipy import spatial
from scipy.optimize import curve_fit
import matplotlib.animation as animation

# Graphs & Analysis

def relative_graph(plot_data, N, cells_arr):
    '''Plots the number cells of each state at each time step as a fraction of all cells'''
    # cells_arr is used instead of x_c and y_c to reduce the number of input  parameters
    fig = plt.figure()
    n_trees = np.zeros(N)     # number of trees
    n_fires = np.zeros(N)     # number of fires
    n_empty = np.zeros(N)     # number of empty cells

    for k in range(N):
        n_trees[k] = np.count_nonzero(plot_data[k]==1)/(cells_arr.shape[0]*cells_arr.shape[1])
        n_fires[k] = np.count_nonzero(plot_data[k]==2)/(cells_arr.shape[0]*cells_arr.shape[1])
        n_empty[k] = np.count_nonzero(plot_data[k]==0)/(cells_arr.shape[0]*cells_arr.shape[1])

    x = np.arange(0,N,1)

    plt.plot(x, n_trees,'g', label='Cells containing trees')
    plt.plot(x, n_fires,'r', label='Cells on fire')
    plt.plot(x, n_empty,'k', label='Empty cells')

    plt.xlabel('Time')
    plt.ylabel('Fraction of all cells')
    plt.legend()
    plt.title('Relative number of cells of each state')
    plt.show()
    
    
def graph_areas(plot_data, state, plot_range):
    '''Plots cluster area against frequency on a log-log graph'''
    
    # plot_range is the range of time stops to draw data from
    # counting cluster sizes at multiple time steps ensures sufficient data for a more accurate plot
    # plot_range must be chosen such that the system has already achieved equilibrium 
    
    c = np.array([], dtype='int')    # array containing all clusters but with extra zeros

    # find clusters and their areas at each time step and save it to c
    for k in plot_range:
        lw, num = ndimage.label(plot_data[k]==state)

        cluster_areas = ndimage.sum_labels(plot_data[k]==state, lw, index=np.arange(lw.max() + 1)).astype('int')

        c = np.concatenate((cluster_areas,c))

    # remove the zeros
    all_areas=c[~(c==0)]    # array containing all clusters

    # create arrays for frequency and size to plot
    count = np.zeros((np.max(all_areas)), dtype=int)
    x = np.arange(0, np.max(all_areas), 1)

    # counts clusters of each size
    for i in x:
        count[i] = np.count_nonzero((all_areas==i))
    
    clr = ['k','g','r']
    
    plt.figure(figsize=[6,5])

    plt.scatter(x, count, c=clr[state], label='N.ipynb_checkpoints\s)')
    
    plt.loglog()
    plt.title('Log-Log plot of cluster area against frequency')
    plt.xlabel('Cluster area / Unit cells')
    plt.ylabel('Frequency')

    plt.show()
    

def graph_areas_fit(plot_data, state, plot_range):
    '''Plots cluster area against frequency on a log-log graph and a line fit'''
    
    # plot_range is the range of time stops to draw data from
    # counting cluster sizes at multiple time steps ensures sufficient data for a more accurate plot
    # plot_range must be chosen such that the system has already achieved equilibrium 
    
    c = np.array([], dtype='int')    # array containing all clusters but with extra zeros

    # find clusters and their areas at each time step and save it to c
    for k in plot_range:
        lw, num = ndimage.label(plot_data[k]==state)

        cluster_areas = ndimage.sum_labels(plot_data[k]==state, lw, index=np.arange(lw.max() + 1)).astype('int')

        c = np.concatenate((cluster_areas,c))

    # remove the zeros
    all_areas=c[~(c==0)]    # array containing all clusters

    # create arrays for frequency and size to plot
    count = np.zeros((np.max(all_areas)), dtype=int)
    x = np.arange(0, np.max(all_areas), 1)

    # counts clusters of each size
    for i in x:
        count[i] = np.count_nonzero((all_areas==i))
    
    plt.figure(figsize=[6,5])
    clr = ['k','g','r']
    plt.scatter(x, count, c=clr[state], label='N(s)')

    x1 = x[~(count==0)]
    count1 = count[~(count==0)]
    b, m = polyfit(np.log10(x1), np.log10(count1), 1)
    plt.plot(x1, 10**b*x1**m, 'b', label='power-law fit')
    
    plt.loglog()
    plt.title('Log-Log plot of cluster area against frequency')
    plt.xlabel('Cluster area / Unit cells')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    return print(f'm = {m}, b = {b}')


def graph_circumference_fit(plot_data, state, n):
    '''Plots distribution of approximate cluster circumferences'''
    # the circumference of a cluster is approximated by finding the smallest circle that encloses the cluster and
    # computing its circumference
    
    c = []   # list containing all cluster centres

    lw, num = ndimage.label(plot_data[n]==state)
    
    # find the centre of each cluster
    cluster_centres = ndimage.center_of_mass(plot_data[n]==state, lw, index=np.arange(lw.max() + 1))
    c.append(cluster_centres)
    
    all_centres=np.array(np.rint(c[~(c==0)][1:]))    # array containing all cluster centres as integers and removes zeros
    
    circumferences = np.zeros(len(all_centres))      # array containing all circumferences
    
    for i in range(len(all_centres)):
        XA = all_centres[i:i+1]       # to avoid XA has to be 2-dimentional error
        
        # finds the distance between the centre and the furthest cell in the cluster from it
        max_distance = np.max(spatial.distance.cdist(XA, np.argwhere(lw == i + 1), metric='euclidean'))
        
        circumferences[i] = 2 * np.pi * max_distance
        
    circumferences = circumferences[~(circumferences==0)].astype('int')
    
    # create arrays for frequency and circumference to plot
    count = np.zeros((np.max(circumferences)), dtype=int)
    x = np.arange(0, np.max(circumferences), 1)

    # counts clusters of each size
    for i in x:
        count[i] = np.count_nonzero((circumferences==i))
    
    plt.figure(figsize=[6,5])
    clr = ['k','g','r']
    plt.scatter(x, count, c=clr[state], label='N(s)')

    x1 = x[~(count==0)]
    count1 = count[~(count==0)]
    b, m = polyfit(np.log10(x1), np.log10(count1), 1)
    plt.plot(x1, 10**b*x1**m, 'b', label='power-law fit')
    
    plt.loglog()
    plt.title('Log-Log plot of cluster circumference against frequency')
    plt.xlabel('Cluster circumference / Unit cells')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    return print(f'm = {m}, b = {b}')

def grid_n_cluster_map(plot_data, state, n):
    '''Create a heat map of clusters by area at time step n'''
    fig = plt.figure()

    lw, num = ndimage.label(plot_data[n]==state)

    cluster_areas0 = ndimage.sum_labels(plot_data[n]==state, lw, index=np.arange(lw.max() + 1)).astype('int')
    plot_clusters0 = cluster_areas0[lw]

    im3 = plt.imshow(plot_clusters0, origin='lower', interpolation='nearest')
    plt.colorbar()
    plt.show()
    

def mean_density(plot_data, state, N_range, cells_arr):
    '''Calculate the mean density of a given state over a range of time steps'''
    # N_range is the range of time steps over which the mean density of a given state is calculated
    # N_range[0] is the satrting point and N_range[1] is the end point
    
    n_trees = np.zeros(len(N_range))     # number of trees
    n_fires = np.zeros(len(N_range))     # number of fires
    n_empty = np.zeros(len(N_range))     # number of empty cells
    
    for k in N_range:
        n_trees[k-N_range[0]] = np.count_nonzero(plot_data[k]==1)/(cells_arr.shape[0]*cells_arr.shape[1])
        n_fires[k-N_range[0]] = np.count_nonzero(plot_data[k]==2)/(cells_arr.shape[0]*cells_arr.shape[1])
        n_empty[k-N_range[0]] = np.count_nonzero(plot_data[k]==0)/(cells_arr.shape[0]*cells_arr.shape[1])
        
    b = np.array([n_empty,n_trees,n_fires])  # Indeces of elements in b correspond to the states they represent
    s = ['empty', 'tree', 'fire']    
    return print(f'The mean density of {s[state]} cells is {np.mean(b[state])}') 

# Animations

def animate(plot_data, an_range, t, loop=True):
    '''Creates an animation of the forest fire simulation using FuncAnimation'''
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = matplotlib.colors.ListedColormap(['Black', 'Green', 'Red'])
    
    # Initialize the plot
    board_image = ax.imshow(plot_data[0], cmap=cmap, vmin=0, vmax=2, origin='lower')

    # The update function that FuncAnimation calls every frame
    def update(frame_index):
        board_image.set_data(plot_data[frame_index])
        return [board_image]

    # t is in seconds for your original code. interval needs milliseconds.
    # Note: 0.00001 seconds is 0.01 ms, which is faster than Matplotlib can render. 
    # It will simply play as fast as your hardware allows.
    interval_ms = max(1, int(t * 1000)) 

    # Create the animation object
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=an_range, 
        interval=interval_ms, 
        blit=True, 
        repeat=loop
    )

    # Show the plot - this keeps the window open and responsive
    plt.show()