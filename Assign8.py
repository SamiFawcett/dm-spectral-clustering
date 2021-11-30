import sys
import numpy as np
import scipy.spatial as spa
import pandas as pd


def degree_matrix(num_nodes, similarity_matrix):
    matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        matrix[i][i] = sum(similarity_matrix[i])

    return matrix


def sortDistances(distances):
    distances.sort(key=lambda x: x[1])
    return distances


def knn_adj_matrix(num_nodes, mutual_knn_edgelist, **kwargs):
    distances = kwargs.get('distances', None)
    if(distances == None):
        distances = np.ones((num_nodes, num_nodes))
    matrix = np.zeros((num_nodes, num_nodes))
    i = 0
    j = 0
    for node in mutual_knn_edgelist:
        for neighbor in mutual_knn_edgelist[node]:
            # default adj matrix is defined with either 1 or 0 in mutual node locations, however the adj matrix can be given edge weights through @param distance
            matrix[node][neighbor] = distances[i][j]
            j += 1
        j = 0
        i += 1

    return matrix


def knn_edgelist(knn_graph):
    mutual_knn = {}
    for node in knn_graph:
        mutual = []
        for node_neighbor in knn_graph[node]:
            if node in knn_graph[node_neighbor]:
                mutual.append(node_neighbor)
        mutual_knn[node] = mutual

    return mutual_knn


def k_nearest_neighbor(points, k):
    k_nearest_neighbors = []
    dist_edge_weights = []
    for i in range(len(points)):
        distances = []
        dew = []
        for j in range(len(points)):
            dist = spa.distance.euclidean(points[i], points[j])
            dew.append(dist)
            if(i == j):
                continue
            else:
                distances.append(
                    (j, dist))

        neighbors = sortDistances(distances)
        dist_edge_weights.append(dew)

        k_nearest_neighbors.append(neighbors[0:k])

    knn_graph = {}
    for i in range(len(k_nearest_neighbors)):
        knn_graph[i] = []
        for neighbors in k_nearest_neighbors[i]:
            knn_graph[i].append(neighbors[0])

    return [knn_graph, dist_edge_weights]


def markov_matrix(knn_adj_mat, deg_mat):
    return np.dot(np.linalg.pinv(deg_mat), knn_adj_mat)


def laplacian_matrix(A, D):
    return D - A


def asym_norm_laplacian_matrix(D, L):
    return np.dot(np.linalg.pinv(D), L)


def sym_norm_laplacian_matrix(D, L):
    neg_sqrt_deg = np.linalg.pinv(np.sqrt(D))
    return np.dot(np.dot(neg_sqrt_deg, L), neg_sqrt_deg)


def normalize_U(U):
    Y = []
    for row in U:
        denom = sum(row)**2
        numer = row
        Y.append(numer/denom)
    return Y


def spectral_clustering(points, k, cut):
    # compute similiarity matrix via knn
    knn, node_distances = k_nearest_neighbor(points, k)

    # edge list format of adj graph
    mutual_knn = knn_edgelist(knn)

    # similarity matrix
    A = knn_adj_matrix(len(points), mutual_knn)

    # degree matrix
    D = degree_matrix(len(points), A)

    #normalized_adj_matrix = markov_matrix(A, D)

    L = laplacian_matrix(A, D)
    B = None
    if(cut == 'ratio'):
        B = L
    elif(cut == 'asymmetric'):
        B = asym_norm_laplacian_matrix(D, L)
    elif(cut == 'symmetric'):
        B = sym_norm_laplacian_matrix(D, L)

    # solve for eigenvalues and eigenvectors
    evals, evecs = np.linalg.eigh(B)

    # take k smallest eigen values and their correspond eigen vectors
    U = []
    for i in range(len(evecs) - 1, len(evecs) - k - 1, -1):
        U.append(evecs[i])

    U = np.array(U)
    # Y normalize rows of U using 16.23 eq
    U_t = np.transpose(U)
    Y = normalize_U(U_t)
    # run k-means on Y to get clusterings c1...ck

    return


def getTrueClusterLabels(applianceEnergyAttributeData):
    clusterLabels = []
    for val in applianceEnergyAttributeData:
        if(val <= 40):
            clusterLabels.append(0)
        elif(val <= 60):
            clusterLabels.append(1)
        elif(val <= 100):
            clusterLabels.append(2)
        else:
            clusterLabels.append(3)
    return clusterLabels


def parse(filename, num_points):
    headers = list(pd.read_csv(filename, nrows=0))
    headers.pop(28)
    headers.pop(0)

    data = pd.read_csv(filename, usecols=headers, nrows=num_points).to_numpy()
    dataT = list(np.transpose(data))

    # get appliance energy labels for performance assessment
    applianceEnergyAttr = dataT[0]
    trueClusterLabels = getTrueClusterLabels(applianceEnergyAttr)

    dataT.pop(0)
    dataT = np.array(dataT)

    return [np.transpose(dataT), dataT, headers]


if __name__ == "__main__":

    argv = sys.argv
    filename = argv[1]
    k = int(argv[2])
    n = int(argv[3])
    spread = float(argv[4])
    clustering_objective = argv[5]

    points, attr_view, headers = parse(filename, n)

    spectral_clustering(points, k, clustering_objective)
