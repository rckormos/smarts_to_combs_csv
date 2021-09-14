import argparse
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist

def transform_coords(coords, ref_coords):
    """Transform coordinates to align with reference coordinates.

    Parameters
    ----------
    coords : np.array [M x N x 3]
        Array of 3D coordinates to be transformed.
    ref_coords : np.array [M x N x 3]
        Array of 3D coordinates against which to align coords.

    Returns
    -------
    aligned_coords : np.array [M x N x 3]
        Array of aligned 3D coordinates.
    rmsds : float [M]
        RMSD values of aligned coordinates to reference coordinates.
    """
    mean_coords = np.mean(coords, axis=1, keepdims=True)
    mean_ref = np.mean(ref_coords, axis=1, keepdims=True)
    cov_matrix = np.matmul(np.transpose(coords - mean_coords,
                                        [0, 2, 1]), ref_coords - mean_ref)
    U, S, Wt = np.linalg.svd(cov_matrix)
    R = np.matmul(U, Wt)
    neg_det = np.where(np.linalg.det(R) < 0)
    if np.any(neg_det):
       Wt[neg_det][-1] *= -1.
       R = np.matmul(U, Wt)
    diff = np.matmul(coords - mean_coords, R) + \
           mean_ref - ref_coords
    rmsds = np.sqrt(np.mean(np.square(diff), axis=(1,2)))
    return np.matmul(coords - mean_coords, R) + mean_ref, rmsds

def cluster_greedy(dists, threshold=0.25):
    """Greedily cluster indices from a distance matrix.

    Parameters
    ----------
    dists : np.array [M x M]
        Array of distances between M objects.
    threshold : float, optional
        Threshold value at which to perform greedy clustering.

    Return
    ------
    clusters : list
        List of numpy arrays containing integer indices of cluster members.
    """
    lt_thresh = (dists < threshold).astype(int)
    n_neighbors = np.sum(lt_thresh, axis=0) - 1
    clusters = []
    clustered = np.zeros(len(dists)).astype(bool)
    for i in np.argsort(n_neighbors)[::-1]:
        if not clustered[i]:
            cluster = np.argwhere(np.logical_and(lt_thresh[i], ~clustered))
            clustered[cluster] = True
            clusters.append(cluster)
    return clusters

def parse_args():
    argp = argparse.ArgumentParser()
    argp.add_argument('csv', help="CSV to cluster.")
    argp.add_argument('-t', '--threshold', type=float, default=0.25, 
                      help="Clusrering threshold (in Angstroms).")
    return argp.parse_args()

if __name__ == "__main__":
    args = parse_args()
    csv_in = args.csv
    df = pd.read_csv(csv_in)
    N = len(set(df['generic_name']))
    pos = np.array(df[['c_x', 'c_y', 'c_z']]).reshape((-1, N, 3))
    M = len(pos)
    idxs0, idxs1 = np.triu_indices(M)
    all_rmsds = []
    for m in range(M - 2):
        pos0 = pos[idxs0[m*(M - 1):(m + 1)*(M - 1)]]
        pos1 = pos[idxs1[m*(M - 1):(m + 1)*(M - 1)]]
        _, rmsds = transform_coords(pos0, pos1)
        all_rmsds.append(rmsds)
    rmsds = np.hstack(all_rmsds)
    # coords_align, rmsds = transform_coords(pos[idxs0], pos[idxs1])
    rmsds_array = np.zeros((M, M))
    rmsds_array[idxs0, idxs1], rmsds_array[idxs1, idxs0] = rmsds, rmsds
    clusters = cluster_greedy(rmsds_array, args.threshold)
    for i, cluster in enumerate(clusters):
        cluster_idxs = np.zeros(len(cluster) * N, dtype=int)
        for j in range(N):
            cluster_idxs[j::N] = N * cluster.flatten() + j
        df_cluster = df.iloc[cluster_idxs]
        csv_name = csv_in[:-4] + '_cluster' + str(i) + '.csv'
        df_cluster.to_csv(csv_name)
