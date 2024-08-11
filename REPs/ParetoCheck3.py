import numpy as np

def ParetoCheck(f1_train, f3_train, f2_train):
    """
    Given a set of points in 3D space, returns the Pareto optimal points.
    
    Args:
        points (numpy.ndarray): A 2D array of shape (n_points, 3) representing the 3D points.
    
    Returns:
        numpy.ndarray: A 2D array of shape (n_frontier_points, 3) representing the Pareto optimal points.
    """
    # Initialize the input
    points = np.hstack([f1_train.reshape(len(f1_train),1), f3_train.reshape(len(f3_train),1), f2_train.reshape(len(f2_train),1)])

    # # Calculate the minimum and maximum values for each axis
    # mins = np.min(points, axis=0)
    # maxs = np.max(points, axis=0)

    # Initialize empty list of Pareto optimal points
    pareto_frontier = []

    # Iterate over points and check if they are Pareto optimal
    for i, point in enumerate(points):
        is_pareto = True
        for j, other_point in enumerate(points):
            if j != i and np.all(other_point <= point):
                is_pareto = False
                break
        if is_pareto:
            pareto_frontier.append(point)
            

    # Convert list to numpy array and return
    pareto_frontier = np.array(pareto_frontier)

    f1_train = pareto_frontier[:,0]
    f3_train = pareto_frontier[:,2]
    f2_train = pareto_frontier[:,1]
    
    return f1_train, f3_train, f2_train
