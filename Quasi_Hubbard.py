# Imports
import sys
from pathlib import Path

import matplotlib.colors as mcolors
from joblib import Parallel, delayed
from scipy.sparse import lil_matrix, load_npz, save_npz

# from functions import compute_eigenstate
from functions import *
from generate_lattice import generate_sites, generate_octagon, clean_rings
from potential_functions import *

# Load style file
plt.style.use('PaperDoubleFig.mplstyle')

if __name__ == "__main__":

    # Define lattice parameters

    lattice_params = dict()
    lattice_params['length'] = 70  # System size in units of lattice wavelength
    lattice_params['depth'] = float(sys.argv[1])  # Lattice depth in units of recoil energy
    lattice_params['cut_off'] = 4.0  # Cut off radius in units of lattice wavelength
    lattice_params['global_step'] = 0.1  # Grid spacing in units of lattice wavelength
    x_global, y_global = generate_grid((0.0, 0.0), lattice_params['length'] / 2 + lattice_params['cut_off'] + 0.5,
                                       lattice_params["global_step"])
    lattice_params['x_global'] = x_global  # Global spatial grid for the entire system
    lattice_params['y_global'] = y_global  # Global spatial grid for the entire system
    lattice_params['n_x'] = len(x_global)  # Number of grid points in x direction

    k0 = 2 * np.pi
    kx = np.array([1, 0]) * k0
    ky = np.array([0, 1]) * k0
    kp = 1 / np.sqrt(2) * (kx + ky)  # 8fold Quasicrystal diagonal beams
    km = 1 / np.sqrt(2) * (kx - ky)  # 8fold Quasicrystal diagonal beams

    ##QC setup
    delta_y = 121 + np.sqrt(5)
    delta_x = 75.0
    phi1 = -2 * np.pi * delta_x
    phi2 = -2 * np.pi * delta_y
    phi3 = -2 * np.pi * 1 / np.sqrt(2) * (
            delta_x + delta_y)
    phi4 = -2 * np.pi * 1 / np.sqrt(2) * (
            delta_x - delta_y)
    k = np.array([kx, ky, kp, km])
    phis = np.array([phi1, phi2, phi3, phi4])  # phases of the lattice laser beams

    lattice_params['k'] = k
    lattice_params['phis'] = phis

    runtic = time.perf_counter()

    # Create directory for saving files
    dirname = "size_%.2f/fine/results_depth_%.4f" % (lattice_params['length'], lattice_params["depth"])
    Path(dirname).mkdir(parents=True, exist_ok=True)
    print(dirname)
    Xglobal, Yglobal = np.meshgrid(x_global, y_global)
    V_global = potential(Xglobal, Yglobal, lattice_params["depth"], k, phis)

    minima = generate_sites(x_global[0], x_global[-1], y_global[0], y_global[-1], (0.0, 0.0), lattice_params,
                            mask_radius=lattice_params['length'] / 2)  # Generate list of minima
    np.save(dirname + '/phis', phis)
    octagon = generate_octagon(minima, phis)  # Generate list of configuration space coordinates of the lattice sites
    ##Eliminate pathological minima that do not host a Wannier function (see Appendix F of PRB paper)
    minima_clean, rings_list = clean_rings(minima, octagon)
    np.save(dirname + "/lattice_sites", minima_clean)
    print("lattice sites saved")
    octagon_clean = generate_octagon(minima_clean, phis)
    np.save(dirname + '/octagon', octagon_clean)
    print("octagon sites saved")
    np.save(dirname + '/rings_list', rings_list)
    n_rings = rings_list.shape[0]
    lattice_params['rings_list'] = rings_list
    lattice_params['lattice_sites'] = minima_clean
    n_states = minima_clean.shape[0]  # + 3*rings_list.shape[0]
    print("sites loaded: " + str(n_states) + " sites")
    cm = 1 / 2.54
    plt.figure(2, figsize=(15 * cm, 15 * cm))
    plt.subplot(1, 1, 1, aspect='equal')
    x_minima_clean, y_minima_clean = zip(*minima_clean)
    x_minima, y_minima = zip(*minima)
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=2 * lattice_params["depth"], vmax=4 * lattice_params["depth"])
    plt.pcolor(x_global, y_global, V_global, cmap='jet', norm=norm, alpha=0.6, edgecolor='None')
    plt.scatter(x_minima, y_minima, s=5, c='k', zorder=1, alpha=0.6
                , edgecolor='None')
    plt.scatter(x_minima_clean, y_minima_clean, s=6, c='r'
                , zorder=3, alpha=0.6, edgecolor='None')
    plt.xlim(np.min(x_minima) - 1.0, np.max(x_minima) + 1.0)
    plt.ylim(np.min(y_minima) - 1.0, np.max(y_minima) + 1.0)
    plt.savefig(dirname + "/lattice_sites_V.png", dpi=600)
    sites_number = n_states
    np.save("sites_number", sites_number)

    tic = time.perf_counter()

    print("Number of sites" + str(n_states))
    n_x = lattice_params['n_x']
    n_y = n_x
    window_radius = lattice_params["cut_off"] + 1.25
    x_window, y_window = generate_grid((0.0, 0.0), window_radius, lattice_params["global_step"])
    n_x_window = len(x_window)
    n_y_window = n_x_window
    # Compute NON-ORTHOGONAL Wannier functions
    if Path(dirname + "/wannier_functions.npy").is_file():
        print('Wannier functions already exist ')
        wannier_functions_vec = np.load(dirname + "/wannier_functions.npy")
    else:
        print("Start generation of Wannier Functions")
        wannier_functions_vec = np.hstack(
            np.array(Parallel(n_jobs=-1, verbose=100, backend='multiprocessing', batch_size=1)(
                delayed(generate_wannier_function)(i_site, lattice_params, plot=False) for i_site in
                range(minima_clean.shape[0]))))
    print("Wannier functions generated")

    ##Compute overlap matrix
    S_matrix = lil_matrix(np.zeros(n_states))
    H_wannier = lil_matrix(np.zeros(n_states))
    if Path(dirname + "/S_matrix.npy").is_file() == True:
        print('H wannier already exist ')
        H_wannier = load_npz(dirname + "/hamiltonian_wannier.npz")
    else:
        H_wannier = scipy.sparse.hstack(np.array(
            Parallel(n_jobs=-1, verbose=100, backend='multiprocessing', batch_size=1)(
                delayed(compute_H)(i_site, wannier_functions_vec, minima_clean, lattice_params["cut_off"]
                                   , window_radius, lattice_params["global_step"],
                                   lattice_params) for i_site in
                range(minima_clean.shape[0]))), format="csc")  ##Only compute lower half of H
        H_wannier = (H_wannier + H_wannier.T)  # makes H hermitian
        save_npz(dirname + "/hamiltonian_wannier", H_wannier)
        print("H wannier generated")

    S_matrix = scipy.sparse.hstack(np.array(
        Parallel(n_jobs=-1, verbose=100, backend='multiprocessing', batch_size=1)(
            delayed(compute_S)(i_site, wannier_functions_vec, minima_clean, lattice_params["cut_off"]
                               , window_radius, lattice_params["global_step"]) for
            i_site in
            range(minima_clean.shape[0]))), format="csc")
    S_matrix = (S_matrix + S_matrix.T)  # makes S symmetric
    S_matrix = S_matrix.todense()
    np.save(dirname + "/S_matrix.npy", S_matrix)
    print("S matrix generated")
    # Compute Lowdin orthogonalization matrix
    S_diag, U_mat = np.linalg.eigh(S_matrix)
    S_sqrt_inv = np.diag(np.sqrt(np.divide(1, S_diag)))
    symm_orthog = S_sqrt_inv.dot(
        np.conj(U_mat.T))
    symm_orthog = U_mat.dot(symm_orthog)
    symm_orthog = np.array(symm_orthog)

    print("Symmetric orthogonalization matrix generated")
    H_lowdin = H_wannier.dot(np.conj(symm_orthog.T))
    H_lowdin = csc_matrix(symm_orthog.dot(H_lowdin))
    save_npz(dirname + '/hamiltonian_real.npz', H_lowdin)
    print("Hamiltonian saved")
    # Compute Lowdin orthogonalized Wannier functions
    if Path(dirname + "/lowdin_basis_vec.npy").is_file():
        print('lowdin WFs already exist ')
        wannier_functions = np.load(dirname + "/lowdin_basis_vec.npy")
    else:
        wannier_functions = np.hstack(np.array(
            Parallel(n_jobs=-1, verbose=100, backend='multiprocessing', batch_size=1)(
                delayed(compute_lowdin)(i_site, wannier_functions_vec, minima_clean, symm_orthog,
                                        window_radius, lattice_params["globa_step"], plot=False) for i_site in
                range(minima_clean.shape[0]))))
        # np.save(dirname+'/lowdin_basis_vec', lowdin_basis_vec)

    np.save(dirname + "/y_window", y_window)
    np.save(dirname + "/x_window", x_window)

    runtoc = time.perf_counter()
    ##Hubbard U: on-site interaction energy
    if Path(dirname + "/hubbard_U.npy").is_file() == False:
        hubbard_U = np.power((abs(wannier_functions)), 4)
        hubbard_U = (np.sum(hubbard_U, axis=0) * lattice_params["global_step"] ** 2)
        np.save(dirname + "/hubbard_U", hubbard_U)
        print("hubbard U computed")
    ## Off-site interactions
    U_iijj = scipy.sparse.hstack(np.array(
        Parallel(n_jobs=-1, verbose=100, backend='multiprocessing', batch_size=1)(
            delayed(compute_U_iijj)(i_site, wannier_functions, minima_clean, lattice_params["cut_off"]
                                    , window_radius, lattice_params["global_step"]) for
            i_site in
            range(minima_clean.shape[0]))), format="csc")
    U_iijj = (U_iijj + U_iijj.T)
    U_iijj = U_iijj.todense()
    np.save(dirname + "/U_iijj.npy", U_iijj)
    print("U_iijj computed")

    U_iiij = scipy.sparse.hstack(np.array(
        Parallel(n_jobs=-1, verbose=100, backend='multiprocessing', batch_size=1)(
            delayed(compute_U_iiij)(i_site, wannier_functions, minima_clean, lattice_params["cut_off"]
                                    , window_radius, lattice_params["global_step"]) for
            i_site in
            range(minima_clean.shape[0]))), format="csc")
    U_iiij = U_iiij.todense()
    np.save(dirname + "/U_iiij.npy", U_iiij)
    print("U_iiij computed")
