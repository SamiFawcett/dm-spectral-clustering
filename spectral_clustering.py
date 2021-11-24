import sys
import numpy as np
import scipy.spatial as spa
import pandas as pd


def degree_matrix(num_nodes, mutual_knn_edgelist):
    matrix = np.zeros((num_nodes, num_nodes))
    i = 0
    for node in mutual_knn_edgelist:
        degree = len(mutual_knn_edgelist[node])
        matrix[i][i] = degree
        i += 1
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
    return np.dot(np.linalg.inv(deg_mat), knn_adj_mat)


def laplacian_matrix(A, D):
    return D - A


def normalized_laplacian_matrix(D, L):
    return np.dot(np.linalg.inv(D), L)


def spectral_clustering(points):
    k = 3
    knn, node_distances = k_nearest_neighbor(points, k)
    # edge list format of adj graph
    mutual_knn = knn_edgelist(knn)
    # distances=node_distances for weighted edges
    A = knn_adj_matrix(len(points), mutual_knn)
    D = degree_matrix(len(points), mutual_knn)

    normalized_adj_matrix = markov_matrix(A, D)
    L = laplacian_matrix(A, D)
    L_normalized = normalized_laplacian_matrix(A, D)  # asymmetric

    print(L)
    return 0


def parse(filename):
    features = list(pd.read_csv(filename, nrows=0))
    point_view = pd.read_csv(filename, usecols=features).to_numpy()
    col_view = np.transpose(point_view)
    return [point_view, col_view]


if __name__ == "__main__":

    argv = sys.argv

    filename = argv[1]
    # generatePoints(n) if want to generate points rather than input points from file

    data = parse(filename)
    points = data[0]
    col_view = data[1]
    print(points)
    spectral_clustering(points)
