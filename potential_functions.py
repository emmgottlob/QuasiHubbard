##Imports
from typing import List, Union, Tuple, Optional

import numpy as np
# from numpy.matrix import transpose
import scipy.optimize


def potential(Xmesh: np.ndarray, Ymesh: np.ndarray, depth: Union[float, int], k: np.ndarray,
              phis: np.ndarray) -> np.ndarray:
    """
    Calculate the potential based on the given mesh grids, depth, wavevectors, and phases.

    Parameters:
        Xmesh (np.ndarray): Mesh grid for the x-coordinate.
        Ymesh (np.ndarray): Mesh grid for the y-coordinate.
        depth (Union[float, int]): Depth parameter for the potential.
        k (np.ndarray): Wavevectors.
        phis (np.ndarray): Phases for each wavevector.

    Returns:
        np.ndarray: Calculated potential.
    """
    res = 0
    for i in range(len(k)):
        res += depth * np.square(np.cos(k[i][0] * Xmesh + k[i][1] * Ymesh + phis[i]))

    return res


def fun_potential(X: np.ndarray, depth: Union[float, int], k: np.ndarray, phis: np.ndarray,
                  mask_on: Optional[bool] = False, mask_radius: Optional[Union[float, int]] = 0) -> Union[float, int]:
    """
    Calculate the potential for a specific point.

    Parameters:
        X (np.ndarray): Coordinates [x, y] of the point.
        depth (Union[float, int]): Depth parameter for the potential.
        k (np.ndarray): Wavevectors.
        phis (np.ndarray): Phases for each wavevector.
        mask_on (Optional[bool]): Whether to apply a mask. Defaults to False.
        mask_radius (Optional[Union[float, int]]): Radius for the mask. Defaults to 0.

    Returns:
        Union[float, int]: Calculated potential at the given point.
    """
    x = X[0]
    y = X[1]
    res_vec = depth * np.square(np.cos(np.multiply(k[:, 0], x) + np.multiply(k[:, 1], y) + phis[:]))
    res = np.sum(res_vec)

    return res


def fun_jacobian(X: np.ndarray, depth: Union[float, int], k: np.ndarray, phis: np.ndarray) -> np.ndarray:
    """
    Compute the Jacobian of the potential function at a given point.

    Parameters:
        X (np.ndarray): Coordinates [x, y] of the point.
        depth (Union[float, int]): Depth parameter for the potential.
        k (np.ndarray): Wavevectors.
        phis (np.ndarray): Phases for each wavevector.

    Returns:
        np.ndarray: Jacobian vector at the given point.
    """
    x = X[0]
    y = X[1]
    res_vec1 = depth * (-1) * k[:, 0] * np.sin(2 * (k[:, 0] * x + k[:, 1] * y + phis[:]))
    res1 = np.sum(res_vec1)
    res_vec2 = depth * (-1) * k[:, 1] * np.sin(2 * (k[:, 0] * x + k[:, 1] * y + phis[:]))
    res2 = np.sum(res_vec2)
    return np.array([res1, res2])


def fun_hessian(X: np.ndarray, depth: Union[float, int], k: np.ndarray, phis: np.ndarray) -> np.ndarray:
    """
    Compute the Hessian matrix of the potential function at a given point.

    Parameters:
        X (np.ndarray): Coordinates [x, y] of the point.
        depth (Union[float, int]): Depth parameter for the potential.
        k (np.ndarray): Wavevectors.
        phis (np.ndarray): Phases for each wavevector.

    Returns:
        np.ndarray: Hessian matrix at the given point.
    """
    x = X[0]
    y = X[1]
    curv_x = diffdiffpotential(x, y, 0, 0, depth, k, phis)
    curv_y = diffdiffpotential(x, y, 1, 1, depth, k, phis)
    curv_xy = diffdiffpotential(x, y, 0, 1, depth, k, phis)
    hess = np.array([[curv_x, curv_xy], [curv_xy, curv_y]])
    return hess


def hessian_mat(x, y, depth, k, phis):
    """
    Compute the Hessian matrix of the potential function at a given point.

    Parameters:
        X (np.ndarray): Coordinates [x, y] of the point.
        depth (Union[float, int]): Depth parameter for the potential.
        k (np.ndarray): Wavevectors.
        phis (np.ndarray): Phases for each wavevector.

    Returns:
        np.ndarray: Hessian matrix at the given point.
    """
    curv_x = diffdiffpotential(x, y, 0, 0, depth, k, phis)
    curv_y = diffdiffpotential(x, y, 1, 1, depth, k, phis)
    curv_xy = diffdiffpotential(x, y, 0, 1, depth, k, phis)
    hess = np.array([[curv_x, curv_xy], [curv_xy, curv_y]])
    return hess


def diffpotential(Xmesh: np.ndarray, Ymesh: np.ndarray, diff_var: int, depth: Union[float, int],
                  k: np.ndarray, phis: np.ndarray) -> Union[float, int]:
    """
    Compute the first partial derivative of the potential at given points in the mesh grid.

    Parameters:
        Xmesh (np.ndarray): X coordinates of the mesh grid.
        Ymesh (np.ndarray): Y coordinates of the mesh grid.
        diff_var (int): Variable with respect to which the derivative is taken (0 for x, 1 for y).
        depth (Union[float, int]): Depth parameter for the potential.
        k (np.ndarray): Wavevectors.
        phis (np.ndarray): Phases for each wavevector.

    Returns:
        Union[float, int]: Value of the first partial derivative at the given points.
    """
    res_vec = depth * (-1) * k[:, diff_var] * np.sin(2 * (k[:, 0] * Xmesh + k[:, 1] * Ymesh + phis[:]))
    res = np.sum(res_vec)
    return res


def diffdiffpotential(Xmesh: np.ndarray, Ymesh: np.ndarray, diff_var1: int, diff_var2: int,
                      depth: Union[float, int], k: np.ndarray, phis: np.ndarray) -> Union[float, int]:
    """
    Compute the second partial derivative of the potential at given points in the mesh grid.

    Parameters:
        Xmesh (np.ndarray): X coordinates of the mesh grid.
        Ymesh (np.ndarray): Y coordinates of the mesh grid.
        diff_var1 (int): First variable with respect to which the derivative is taken (0 for x, 1 for y).
        diff_var2 (int): Second variable with respect to which the derivative is taken (0 for x, 1 for y).
        depth (Union[float, int]): Depth parameter for the potential.
        k (np.ndarray): Wavevectors.
        phis (np.ndarray): Phases for each wavevector.

    Returns:
        Union[float, int]: Value of the second partial derivative at the given points.
    """
    res_vec = depth * (-2) * k[:, diff_var1] * k[:, diff_var2] * np.cos(
        2 * (k[:, 0] * Xmesh + k[:, 1] * Ymesh + phis[:]))
    res = np.sum(res_vec)
    return res


def potential_mask(Xmesh: np.ndarray, Ymesh: np.ndarray, V_mat: np.ndarray, center: Tuple[float, float],
                   radius: Union[float, int], depth: Union[float, int], k: np.ndarray, phis: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Compute the potential with a masking effect, setting potential values outside a given radius to a constant value.

    Parameters:
        Xmesh (np.ndarray): X coordinates of the mesh grid.
        Ymesh (np.ndarray): Y coordinates of the mesh grid.
        V_mat (np.ndarray): Potential matrix.
        center (Tuple[float, float]): Coordinates of the center of the mask.
        radius (Union[float, int]): Radius of the masking circle.
        depth (Union[float, int]): Depth parameter for the potential.
        k (np.ndarray): Wavevectors.
        phis (np.ndarray): Phases for each wavevector.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Updated potential matrix with masking effect, Boolean mask matrix.
    """
    mask = (Xmesh - center[0]) ** 2 + (Ymesh - center[1]) ** 2 >= radius ** 2
    V_mask = np.copy(V_mat)
    V_mask[mask] = len(k) * depth
    return V_mask, mask


def potential_quasi1D(Xmesh, Ymesh, depth, k, phis):
    res = 0
    for i in range(len(k)):
        # res+= depth*np.square((np.cos((k[i][0]*Xmesh+k[i][1]*Ymesh)/2+config.phis[i])))
        res += depth * np.square(np.cos(k[i][0] * Xmesh + k[i][1] * Ymesh + phis[i]))
    mask = np.abs(Ymesh) >= 0.25
    V_mask = np.copy(res)
    V_mask[mask] = len(k) * depth
    # res = amp*((np.cos(np.dot(kx,[x,y])/2+phi1))**2+(np.cos(np.dot(ky,[x,y])/2+phi2))**2+(np.cos(np.dot(kp,[x,y])/2+phi3))**2+(np.cos(np.dot(km,[x,y])/2+phi4))**2)
    return V_mask


def potential_mask_site(Xmesh: np.ndarray, Ymesh: np.ndarray, V_mat: np.ndarray, site: Tuple[float, float],
                        depth: Union[float, int], k: np.ndarray, phis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the potential mask for a specific site, taking into account the local shape of the potential landscape.

    Parameters:
        Xmesh (np.ndarray): X coordinates of the mesh grid.
        Ymesh (np.ndarray): Y coordinates of the mesh grid.
        V_mat (np.ndarray): Potential matrix.
        site (Tuple[float, float]): Coordinates of the site for which the mask is calculated.
        depth (Union[float, int]): Depth parameter for the potential.
        k (np.ndarray): Wavevectors.
        phis (np.ndarray): Phases for each wavevector.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Updated potential matrix with site-specific masking, Boolean mask matrix.
    """
    # Some constants
    alpha = 2.5
    coef = 0.9
    mask_radius = 0.35

    # Calculate the Hessian matrix and its eigenvalues and eigenvectors
    h_mat = hessian_mat(site[0], site[1], depth, k, phis)
    li, U = np.linalg.eig(h_mat)
    li = np.abs(li[np.argsort(li)])
    U = U[:, np.argsort(li)]

    # Find the angle and lengths based on the eigenvectors and eigenvalues
    theta = np.arctan2(U[0][1], U[0][0])
    w1 = np.sqrt(np.max(li))
    w2 = np.sqrt(np.min(li))
    length_ratio = np.sqrt(w2 / w1)

    # Create the mask and apply it to the potential matrix
    V_mask, mask = potential_mask_contour(Xmesh, Ymesh, V_mat, site, alpha, coef * length_ratio * mask_radius,
                                          coef * mask_radius, -theta, depth, k, phis)

    return V_mask, mask


def potential_mask_contour(Xmesh: np.ndarray, Ymesh: np.ndarray, V_mat: np.ndarray, pos: Tuple[float, float],
                           alpha: Union[float, int], r1: Union[float, int], r2: Union[float, int],
                           theta: Union[float, int], depth: Union[float, int],
                           k: np.ndarray, phis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the masked potential considering both contour and elliptical shape of the mask.

    Parameters:
        Xmesh (np.ndarray): X coordinates of the mesh grid.
        Ymesh (np.ndarray): Y coordinates of the mesh grid.
        V_mat (np.ndarray): Potential matrix.
        pos (Tuple[float, float]): Coordinates of the site for which the mask is calculated.
        alpha (Union[float, int]): Scaling factor for mask contour.
        r1 (Union[float, int]): Semi-major axis of the ellipse.
        r2 (Union[float, int]): Semi-minor axis of the ellipse.
        theta (Union[float, int]): Angle of rotation for the ellipse.
        depth (Union[float, int]): Depth parameter for the potential.
        k (np.ndarray): Wavevectors.
        phis (np.ndarray): Phases for each wavevector.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Updated potential matrix with contour-based masking, Boolean mask matrix.
    """

    # Contour masking
    mask_contour = potential((Xmesh - pos[0]) / alpha + pos[0], (Ymesh - pos[1]) / alpha + pos[1], depth, k, phis) \
                   <= potential(pos[0], pos[1], depth, k, phis) + 0.15 * depth * len(k)

    # Ellipse masking
    mask_ellipse = (np.square((Xmesh - pos[0]) * np.cos(theta) + (Ymesh - pos[1]) * np.sin(theta)) / (1.1 * r1) ** 2
                    + np.square((Xmesh - pos[0]) * np.sin(theta) - (Ymesh - pos[1]) * np.cos(theta)) / (
                            1.1 * r2) ** 2) <= 1

    V_mask = np.copy(V_mat)
    mask = np.logical_and(mask_contour, mask_ellipse)
    mask = 1 - mask
    V_mask[mask] = len(k) * depth

    return V_mask, mask


def potential_mask_hull(Xmesh: np.ndarray, Ymesh: np.ndarray, V_mat: np.ndarray, sites: np.ndarray,
                        radius: float, depth: Union[float, int], k: np.ndarray, phis: np.ndarray,
                        rings_list: List[np.ndarray] = [], hull_flag: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the masked potential based on convex hulls around given sites.

    Parameters:
        Xmesh (np.ndarray): X coordinates of the mesh grid.
        Ymesh (np.ndarray): Y coordinates of the mesh grid.
        V_mat (np.ndarray): Potential matrix.
        sites (np.ndarray): Coordinates of lattice sites.
        radius (float): Radius for masking around the lattice sites.
        depth (Union[float, int]): Depth parameter for the potential.
        k (np.ndarray): Wavevectors.
        phis (np.ndarray): Phases for each wavevector.
        rings_list (List[np.ndarray], optional): List of ring-like formations, if any.
        hull_flag (bool, optional): Flag to indicate if convex hull masking is to be applied.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Updated potential matrix with contour-based masking, Boolean mask matrix.
    """
    masks = []
    hull_object = scipy.spatial.ConvexHull(sites)
    hull = sites[hull_object.vertices, :]
    hull_center = np.array([np.mean(hull[:, 0]), np.mean(hull[:, 1])])
    hull_radii = np.sqrt(np.square(hull[:, 0] - hull_center[0]) + np.square(hull[:, 1] - hull_center[1]))
    hull_enlarged = hull + (np.divide(hull - hull_center, np.vstack([hull_radii, hull_radii]).T)) * 0.17
    points = np.zeros((len(Xmesh.ravel()), 2))
    points[:, 0] = Xmesh.ravel()
    points[:, 1] = Ymesh.ravel()
    mask_hull = in_hull(points, hull_enlarged)
    mask_hull = mask_hull.reshape(Xmesh.shape)
    mask_multiple = mask_hull
    for i in range(len(sites)):
        site = sites[i, :]
        V_mask, mask = potential_mask_site(Xmesh, Ymesh, V_mat, site, depth, k,
                                           phis)
        masks.append(mask)

    if len(rings_list) != 0:
        for i_ring in range(len(rings_list)):
            ring_radius = 0.5
            V_mask_ring, mask_ring = potential_mask(Xmesh, Ymesh, V_mat, rings_list[i_ring], ring_radius, depth, k,
                                                    phis)
            masks.append(mask_ring)
    for i in range(len(masks)):
        mask_multiple = np.logical_and(mask_multiple, masks[i])
    V_mask = np.copy(V_mat)
    V_mask[mask_multiple] = len(k) * depth
    return V_mask, mask_multiple


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) < 0
