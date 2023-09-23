# Functions handling the generation of lattice sites around local lattice potential minima
# Imports
from typing import Union, Tuple, Dict

import numpy as np
# from numpy.matrix import transpose
import scipy.optimize
from scipy.cluster.hierarchy import single, fcluster
from scipy.ndimage.filters import minimum_filter
from scipy.spatial.distance import pdist

import functions
from potential_functions import potential, fun_potential, fun_jacobian, fun_hessian


def generate_sites(center: Tuple[float, float], lattice_params: Dict[str, float],
                   mask_radius: float = 0, mask_radius_y: float = 0,
                   mask_radius_x: float = 0) -> np.ndarray:
    """
    Generate lattice sites based on potential minima within a specified region.

    Parameters:
        center (Tuple[float, float]): The coordinates of the center point.
        lattice_params (Dict[str, float]): Parameters for the lattice like depth, k, and phis.
        mask_radius (float, optional): Restriction on the distance of points from the center.
        mask_radius_y (float, optional): Restriction on the y-coordinate distance from the center.
        mask_radius_x (float, optional): Restriction on the x-coordinate distance from the center.

    Returns:
        np.ndarray: A list of coordinates for the lattice sites.
    """
    sites_list = np.zeros((0, 2))
    depth = lattice_params['depth']
    k = lattice_params['k']
    phis = lattice_params['phis']
    x_array, y_array = functions.generate_grid(center, lattice_params['length'] / 2 + lattice_params['cut_off'] + 0.5,
                                               0.03 / 5)
    X, Y = np.meshgrid(x_array, y_array)
    V = potential(X, Y, depth, k, phis)
    minima = minimum_filter(V, size=3, mode='constant', cval=0.0) == V
    Xvals = X[np.where(minima)]
    Yvals = Y[np.where(minima)]
    rough_sites_list = np.zeros((len(Xvals), 2))
    rough_sites_list[:, 0] = Xvals
    rough_sites_list[:, 1] = Yvals

    for i_site in range(len(Xvals)):
        pos = rough_sites_list[i_site, :]
        radius = np.sqrt(np.square(pos[0] - center[0]) + np.square(pos[1] - center[1]))
        if radius <= mask_radius:
            res = scipy.optimize.minimize(fun_potential, pos, args=(depth, k, phis), method="Newton-CG",
                                          jac=fun_jacobian, hess=fun_hessian)
            hessian = fun_hessian(res.x, depth, k, phis)
            hessian_eigval, eigvecs = np.linalg.eig(hessian)
            check_saddle = np.all(hessian_eigval > 0)
            distance = np.sqrt(np.square(res.x[0] - center[0]) + np.square(res.x[1] - center[1]))
            distance_x = np.abs(res.x[0] - center[0])
            distance_y = np.abs(res.x[1] - center[1])
            cond_x = True
            cond_y = True
            if mask_radius_x != 0:
                cond_x = np.abs(distance_x) < mask_radius_x

            if mask_radius_y != 0:
                cond_y = np.abs(distance_y) < mask_radius_y
            if np.abs(distance) <= mask_radius and check_saddle and cond_x and cond_y:
                sites_list = np.vstack([sites_list, res.x])

    sites_list_clean = clean_lattice_sites(
        np.array(sites_list).reshape((len(sites_list), 2)))  # Remove doublons close sites

    return sites_list_clean


def clean_lattice_sites(minima_list: np.ndarray) -> np.ndarray:
    """
    Cleans up lattice sites by merging close points into centroids.

    Parameters:
        minima_list (np.ndarray): A list of coordinates representing lattice sites.

    Returns:
        np.ndarray: A cleaned-up list of coordinates for the lattice sites.
    """
    cut_off = 0.004  # Fine-tune this parameter

    # Compute the pairwise distances and perform hierarchical clustering
    y = pdist(minima_list)
    Z = single(y)
    clusters = fcluster(Z, t=cut_off, criterion='distance')

    # Number of clusters
    n_clusters = np.max(clusters)

    # Initialize array to hold centroids
    clean_minima = np.zeros((n_clusters, 2))

    for i_site in range(n_clusters):
        # Extract the points belonging to the same cluster
        sites_to_merge = minima_list[clusters == (1 + i_site)]

        # Compute the centroid of these points
        x_coord, y_coord = zip(*sites_to_merge)
        centroid = np.array([np.mean(x_coord), np.mean(y_coord)])

        # Store the centroid
        clean_minima[i_site, :] = centroid

    return np.array(clean_minima)


def distance_to_octagon_edge(octagon_points: np.ndarray) -> np.ndarray:
    """
    Calculates the distances from given points to each of the 8 edges of an octagon.

    Parameters:
        octagon_points (np.ndarray): A 2D array where each row is a coordinate (x, y) of a point.

    Returns:
        np.ndarray: A 2D array where each row represents the distances of a point to each of the 8 edges
                    of the octagon. The array has dimensions (number of points, 8).
    """
    distances = np.zeros((octagon_points.shape[0], 8))

    for i_angle in range(8):
        theta = -i_angle * np.pi / 4  # Rotation angle
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))  # Rotation matrix

        # Rotate points
        rotated_points = R.dot(octagon_points.T).T

        # Calculate distance to the edge of octagon
        distances[:, i_angle] = np.pi / 2 - rotated_points[:, 0]

    return distances


def in_octagon(octagon_points: np.ndarray) -> np.ndarray:
    """
    Determines whether given points are inside an octagon.

    Parameters:
        octagon_points (np.ndarray): A 2D array where each row is a coordinate (x, y) of a point.

    Returns:
        np.ndarray: A 1D array of boolean values, where each value corresponds to whether
                    the corresponding point is inside the octagon.
    """
    return np.all(distance_to_octagon_edge(octagon_points) > 0, axis=1)


def clean_rings(minima_list: np.ndarray, octagon_list: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cleans the given list of minima points by removing points that are not inside the octagon.

    Parameters:
        minima_list (np.ndarray): A 2D array where each row is a coordinate (x, y) representing
                                  a local minimum.
        octagon_list (np.ndarray): A 2D array where each row is a coordinate (x, y) representing
                                   a point in an octagon.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                                        1. A cleaned array of minima points (clean_minima).
                                        2. An array of points representing rings (rings_list),
                                           which is empty for now.
    """
    clean_minima = np.copy(minima_list)
    in_octagon_points = in_octagon(octagon_list)
    clean_minima = np.delete(clean_minima, np.argwhere(in_octagon_points == False), axis=0)
    rings_list = np.zeros((0, 2))

    return clean_minima, rings_list


def generate_octagon(site_list: Union[np.ndarray, Tuple[float, float]],
                     phis: Union[np.ndarray, Tuple[float, float, float, float]],
                     ring: bool = False) -> np.ndarray:
    """
    Generates the configuration space coordinates of the 8QC lattice sites.

    Parameters:
        site_list (List[Tuple[float, float]]): A list of tuples representing the coordinates of lattice sites.
        phis (Tuple[float, float, float, float]): A tuple containing four phase angles phi1, phi2, phi3, and phi4.
        ring (bool, optional): If True, function considers the ring. Defaults to False.

    Returns:
        np.ndarray: A 2D array containing the coordinates of the octagon points.
    """
    phi1, phi2, phi3, phi4 = phis
    k0 = 2 * np.pi
    X, Y = zip(*site_list)
    X, Y = np.array(X), np.array(Y)
    theta1 = np.mod(k0 * X + phi1, np.pi) - np.pi / 2
    theta2 = np.mod(k0 * Y + phi2, np.pi) - np.pi / 2
    theta3 = np.mod((k0 * (X + Y)) / np.sqrt(2) + phi3, np.pi) - np.pi / 2
    theta4 = np.mod((k0 * (X - Y)) / np.sqrt(2) + phi4, np.pi) - np.pi / 2
    phi_vector_x = theta1 - (theta3 + theta4) / np.sqrt(2)
    phi_vector_y = theta2 - (theta3 - theta4) / np.sqrt(2)
    octagon = np.zeros((len(phi_vector_x), 2))
    octagon[:, 0] = phi_vector_x
    octagon[:, 1] = phi_vector_y

    return octagon
