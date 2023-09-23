from cont_schrod import *


def min_spread(state_set: np.ndarray, x_mean: float, y_mean: float, x_array: np.ndarray, y_array: np.ndarray,
               i_state: int = 0) -> np.ndarray:
    """
    Computes the state with minimal spread out of a linear combination of states.

    Parameters:
        state_set (np.ndarray): A multi-dimensional array representing the set of states.
        x_mean (float): The mean value of the x-coordinates.
        y_mean (float): The mean value of the y-coordinates.
        x_array (np.ndarray): An array of x-coordinates.
        y_array (np.ndarray): An array of y-coordinates.
        i_state (int, optional): Index of the state for which minimal spread is to be calculated. Defaults to 0.

    Returns:
        np.ndarray: The vector representing the minimal spread of states for the given state index.
    """
    R_matrix = compute_R_matrix(state_set, x_mean, y_mean, x_array, y_array)
    val, vec = np.linalg.eigh(R_matrix)
    val = val[np.argsort(val)]
    vec = vec[:, np.argsort(val)]
    min_vec = vec[:, i_state]
    norm = np.sum(np.square(np.abs(min_vec)))
    min_vec = min_vec / np.sqrt(norm)

    return min_vec


def compute_R_matrix(state_set: np.ndarray, x_mean: float, y_mean: float, x_array: np.ndarray,
                     y_array: np.ndarray) -> np.ndarray:
    """
    Computes the R matrix, defined as <psi_i | (r - r_mean)^2 | psi_j>, where r is the position vector and psi_i/j are states.

    Parameters:
        state_set (np.ndarray): A multi-dimensional array representing the set of states.
        x_mean (float): The mean value of the x-coordinates.
        y_mean (float): The mean value of the y-coordinates.
        x_array (np.ndarray): An array of x-coordinates.
        y_array (np.ndarray): An array of y-coordinates.

    Returns:
        np.ndarray: The computed R matrix.

    Notes:
        The function calculates the R matrix using the provided state set and mean coordinates.
        It uses meshgrid to create X and Y arrays and performs matrix multiplication to compute the R matrix.
    """
    # Create a mesh grid from the x and y arrays
    X, Y = np.meshgrid(x_array, y_array)

    # Compute the differences for x and y arrays
    dx, dy = np.diff(x_array)[0], np.diff(y_array)[0]

    # Compute radii
    radii = (np.square(X - x_mean) + np.square(Y - y_mean))
    radii_vec = np.transpose(radii.flatten())

    # Compute R_matrix
    R_matrix = np.multiply(radii_vec.reshape((len(radii_vec), 1)), (state_set))
    R_matrix = np.matmul(np.conj((state_set.T)), R_matrix) * dx * dy

    return R_matrix
