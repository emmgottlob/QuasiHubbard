# Quasi_Hubbard

Quasi_Hubbard is a Python-based package for generating the Wannier functions and Hubbard Hamiltonians of quasicrystalline potentials. The Wannier functions are constructed using the method developed in [1]. The script currently handles the case of the eightfold optical quasicrystal only, but addition optical quasicrystals will soon be added in the features.

Energies are expressed in recoil energy $E_{rec} = \hbar^2 k^2 / (2 m)$, and distances in units of the optical lattice wavelength $\lambda$, and $k = 2\pi / \lambda$. The eightfold quasicrystalline optical potential is given by:

$$V(\mathbf{r}) = V_0 \sum_{i=1}^4 \cos^2{(\mathbf{k}\cdot \mathbf{r} + \phi_i  )} $$   

If you wish to use results produced with this package in a scientific publication, please cite: 

[1] Hubbard models for quasicrystalline potentials. E. Gottlob and U. Schneider. Phys. Rev. B 107, 144202.

## Requirements

- Python 3.x
- SciPy
- Matplotlib
- Joblib

## Quick Start

You can execute the script by running the following command:

```
python Quasi_Hubbard.py [lattice depth] [system_diameter]
```

Replace `[lattice depth]` with the desired lattice depth in units of recoil energy, and `[system_diameter]` is the desired system radius in units of the wavelength $\lambda$.

## How It Works

### Define Lattice Parameters

The script starts by defining several parameters for the lattice system. It includes variables like system size, lattice depth, grid spacing, and other physical constants.

### File Management

The script saves results and temporary data in a structured directory hierarchy, facilitating better organization and easier retrieval of computational results.

### Parallel Computations

For computationally intensive tasks such as constructing the Wannier functions, computing the Hamiltonian matrix elements, overlap matrix, and off-site interactions calculations, the script leverages parallel computing for improved efficiency.

## Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests or raise issues.

## Contact

If you encounter any issues or have questions, please create an issue or contact the author directly at emm.gottlob@gmail.com

## License

This project is licensed under the MIT License.

## Authors

- Emmanuel Gottlob, contact: emm.gottlob@gmail.com
- Ulrich Schneider, contact: uws20@cam.ac.uk

