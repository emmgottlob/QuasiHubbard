# Handles the Finite-Difference and Sinc discrete variable discretisation of the Schrodinger equation.
import time
from math import pi
from typing import Tuple

import numpy as np
from scipy.sparse import lil_matrix, diags


def index2_to_index1(i_x: int, j_y: int, nx: int) -> int:
    """
    Convert a 2D index (i_x, j_y) to a 1D index.

    Parameters:
    - i_x: The x-coordinate in the 2D grid.
    - j_y: The y-coordinate in the 2D grid.
    - nx: The number of grid points along the x-axis.

    Returns:
    - int: The 1D index corresponding to the given 2D index.
    """
    index = i_x + nx * j_y
    return index


def index1_to_index2(index1: int, nx: int) -> Tuple[int, int]:
    """
    Convert a 1D index to a 2D index (i_x, j_y).

    Parameters:
    - index1: The 1D index in the flattened array.
    - nx: The number of grid points along the x-axis in the 2D grid.

    Returns:
    - Tuple[int, int]: The 2D index (i_x, j_y) corresponding to the given 1D index.
    """
    j_y = np.floor_divide(index1, nx)
    i_x = np.remainder(index1, nx)
    return (i_x, j_y)


def laplacian(nx, ny, dx, dy):
    # Generate the discretized Laplacian Matrix using a 9 point stencil. dx must == dy.
    laplace = lil_matrix((nx * ny, nx * ny))
    n = nx * ny
    for j_y in range(ny):
        for i_x in range(nx):
            index = index2_to_index1(i_x, j_y, nx)
            indices = [(i_x + 1, j_y, nx), (i_x - 1, j_y, nx)]
            for ind in indices:
                if ind[0] < nx and ind[0] > 0 and ind[1] < ny and ind[1] > 0:
                    index2 = index2_to_index1(ind[0], ind[1], nx)
                    laplace[index, index2] = 0.5 / dx ** 2
            indices = [(i_x, j_y + 1, nx), (i_x, j_y - 1, nx)]
            for ind in indices:
                if ind[0] < nx and ind[0] > 0 and ind[1] < ny and ind[1] > 0:
                    index2 = index2_to_index1(ind[0], ind[1], nx)
                    laplace[index, index2] = 0.5 / dy ** 2
            indices = [(i_x + 1, j_y + 1, nx), (i_x - 1, j_y - 1, nx), (i_x - 1, j_y + 1, nx), (i_x + 1, j_y - 1, nx)]
            for ind in indices:
                if ind[0] < nx and ind[0] > 0 and ind[1] < ny and ind[1] > 0:
                    index2 = index2_to_index1(ind[0], ind[1], nx)
                    laplace[index, index2] = 0.25 / (dx * dy)
            laplace[index, index] = -3 / (dx * dy)
    return laplace


def laplacian_DVR(nx: int, ny: int, dx: float, dy: float) -> lil_matrix:
    """
    Generate the Laplacian matrix using the Sinc Discrete Variable Representation (DVR).

    Parameters:
    - nx (int): Number of grid points along the x-axis.
    - ny (int): Number of grid points along the y-axis.
    - dx (float): Grid spacing along the x-axis.
    - dy (float): Grid spacing along the y-axis.

    Returns:
    - laplace (lil_matrix): The discretized Laplacian matrix.

    The positions are indexed as follows: (x_i, y_j) --> m = i + j * nx
    """

    laplace = lil_matrix((nx * ny, nx * ny))  # Initialize an empty sparse matrix
    n = nx * ny  # Total number of grid points

    for i_y in range(ny):
        for i_x in range(nx):
            index_i = index2_to_index1(i_x, i_y, nx)  # Convert 2D index to 1D
            laplace[index_i, index_i] = -np.pi ** 2 / 3 / (dx ** 2) * 2  # Diagonal element

            for j_x in range(nx):
                if j_x != i_x:
                    index_j = index2_to_index1(j_x, i_y, nx)
                    laplace[index_i, index_j] += -2 * (-1) ** (i_x - j_x) / (dx ** 2) / (
                            i_x - j_x) ** 2  # Off-diagonal elements (x direction)

            for j_y in range(ny):
                if j_y != i_y:
                    index_j = index2_to_index1(i_x, j_y, nx)
                    laplace[index_i, index_j] += -2 * (-1) ** (i_y - j_y) / (dy ** 2) / (
                            i_y - j_y) ** 2  # Off-diagonal elements (y direction)

    return laplace


def hamiltonian(x_array: np.ndarray, y_array: np.ndarray, V_mat: np.ndarray) -> (csc_matrix, csc_matrix, csc_matrix):
    """
    Generate the discretized Hamiltonian H = -1/(4pi^2) * p^2 + V(x, y),
    written in dimensionless units (E_rec=1, lambda=1).
    Note that dx must be equal to dy in this version.

    Parameters:
    - x_array (np.ndarray): 1D array of x positions.
    - y_array (np.ndarray): 1D array of y positions.
    - V_mat (np.ndarray): 2D array representing the potential energy V(x, y).

    Returns:
    - H (csc_matrix): The discretized Hamiltonian matrix.
    - laplace (csc_matrix): The discretized Laplacian matrix.
    - V (csc_matrix): The diagonal potential energy matrix.

    The positions are indexed as follows: (x_i, y_j) --> m = i + nx * j
    """

    tic = time.perf_counter()
    nx = len(x_array)
    ny = len(y_array)
    dx = x_array[1] - x_array[0]
    dy = y_array[1] - y_array[0]

    H = lil_matrix((nx * ny, nx * ny))  # Initialize the Hamiltonian matrix
    laplace = laplacian_DVR(nx, ny, dx, dy)  # Compute the Laplacian matrix using DVR

    V_diag = V_mat.flatten()
    V = diags(V_diag, offsets=0, shape=(nx * ny, nx * ny), format="csc", dtype=None)  # Diagonal potential energy matrix

    H = -1 / (4 * pi ** 2) * laplace + V  # Combine to form the Hamiltonian
    toc = time.perf_counter()

    return H.tocsc(), laplace.tocsc(), V.tocsc()
