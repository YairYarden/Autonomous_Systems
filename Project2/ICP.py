from sklearn.neighbors import KDTree, NearestNeighbors
import numpy as np

def assign_closest_pairs_kdtree(pc_A, pc_B):
    """Assign closest points into pairs of point clouds A and B via K-D Tree
    return the indices of point cloud B."""

    # create K-D Tree
    tree = KDTree(pc_B)  # TODO: use KDTREE function

    # find the indices of point cloud B which are closest to point cloud A
    # use query function, and return only 1 nearest neighbor.
    dist, indices = tree.query(pc_A, k=1)  # TODO (use hint: tree.query)

    return indices.ravel(), tree, dist.squeeze()


def estimate_transform(pc_A, pc_B):
    """Estimate the transformation from point cloud B to A using SVD.
    Find rotation matrix R and translation vector t.
    Calculate the error"""

    # shift point clouds to the center
    mean_A = np.mean(pc_A, axis=0)  # TODO
    mean_B = np.mean(pc_B, axis=0)  # TODO

    # center point clouds
    centered_pc_A = pc_A - mean_A  # TODO
    centered_pc_B = pc_B - mean_B  # TODO

    # apply SVD
    W = np.dot(centered_pc_B.T, centered_pc_A)
    U, S, V_T = np.linalg.svd(W)

    # W = np.dot(centered_pc_A.T, centered_pc_B) # TODO
    # U, S, V_T = np.linalg.svd(W) # TODO

    # calculate rotation and translation matrices
    # R = np.dot(U, V_T) # TODO
    R = np.dot(V_T.T, U.T)
    t = mean_A - np.dot(R, mean_B)  # TODO

    return R, t

def transform_pc(R, t, pc):
    "Apply R and t on PC"
    return np.dot(pc, R.T) + t # TODO

def update_total_transformation(R, t, curr_R, curr_t):
    "update rotation and translation matrices"
    tot_R = np.dot(R, curr_R) # TODO
    tot_t = np.dot(R, curr_t) + t # TODO
    return tot_R, tot_t





def icp(pc_A, pc_B, assign_closest_pairs_func, converge_th = 0.001, max_iters = 50):
    "Iterative closest point algorithm main routine"
    # copy pc of B
    curr_pc_B = pc_B.copy()
    pcs_B_for_animation = [curr_pc_B.copy()]

    # initialize total transformation
    R_total = np.eye(3)  # TODO
    t_total = np.zeros((3,))  # TODO

    # init errors
    prev_err = np.inf
    errors = []
    # iterative process
    for i in range(1, max_iters):

        # determine correspondance
        indices, tree, dist = assign_closest_pairs_func(pc_A, curr_pc_B)  # TODO

        pc_target = pc_A
        pc_source = curr_pc_B[indices, :]
        # find transformation
        curr_R, curr_t = estimate_transform(pc_target, pc_source)  # TODO

        # update current source
        curr_pc_B = transform_pc(curr_R, curr_t, curr_pc_B)  # TODO , shift point cloud B to A
        pcs_B_for_animation.append(curr_pc_B.copy())  # TODO

        # update total transformation
        R_total, t_total = update_total_transformation(R_total, t_total, curr_R, curr_t)  # TODO // update R and t

        # Cacl RMSE
        curr_err = np.sqrt(np.mean(dist ** 2))  # TODO
        # check error:

        errors.append(curr_err)
        if np.abs(prev_err - curr_err) < converge_th:  # TODO // run ICP until R and T has not affect anymore
            break
        prev_err = curr_err

    return i, errors, pcs_B_for_animation, R_total, t_total

def filter_pc(pc, min_height = -1):
    "return filtered point cloud - only points above the minimal height"
    pc_filtered = pc.copy()
    indices = pc_filtered[:, 2] > min_height
    return pc_filtered[indices]


def assign_closest_pairs_knn(pc_A, pc_B):
    """Assign closest points into pairs of point clouds A and B via nearest neighbor using K-D Tree fucntion
    return the indices of point cloud B."""

    # create K-D Tree
    tree = KDTree(pc_B, leaf_size=10) # TODO: use KDTREE function for Brute force (similar no NN), hint: change value of leaf_size parameter

    # find the indices of point cloud B which are closest to point cloud A
    # use query function, and return only 1 nearest neighbor.
    dist, indices = tree.query(pc_A, k=1) # TODO

    return indices.ravel(), tree, dist