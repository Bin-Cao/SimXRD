################################################################
# MatgenKit.py
################################################################
from pymatgen.core.structure import Structure, Lattice
import spglib
from ase import Atoms
from pymatgen.analysis.diffraction import xrd
import numpy as np
from .funs import *

def _atom2str(atoms, deformation,lattice_extinction_ratio,lattice_torsion_ratio):
    """
    Convert an ASE Atoms object into a Pymatgen Structure object after applying deformation.

    Parameters:
    atoms (ASE Atoms object): The atomic structure to be converted.
    deformation (ndarray): The deformation matrix to apply.

    Returns:
    Structure: A Pymatgen Structure object containing lattice information, 
               chemical symbols, and scaled positions.
    """
    # Convert the atomic structure to a conventional cell after applying deformation
    _, _, c_atom = prim2conv(atoms, deformation,lattice_extinction_ratio,lattice_torsion_ratio)
    
    # Extract lattice parameters from the conventional cell
    cell = c_atom.get_cell()
    
    # Extract chemical symbols and atomic positions (scaled)
    symbols = c_atom.get_chemical_symbols()
    positions = c_atom.get_scaled_positions()
    
    # Create a Pymatgen Lattice object using the cell parameters
    lattice = Lattice(cell)
    
    # Create and return a Pymatgen Structure object
    return Structure(lattice, symbols, positions)

def get_diff(atom,deformation,lattice_extinction_ratio,lattice_torsion_ratio):
    # atom defined in ASE database
    calculator = xrd.XRDCalculator()
    struc = _atom2str(atom,deformation,lattice_extinction_ratio,lattice_torsion_ratio) 
    
    pattern = calculator.get_pattern(struc, two_theta_range=(10, 80))
    # Due to the limitations of the package, a slight approximation is introduced here. 
    # The peak position is determined according to the set precision
    return pattern.x, pattern.y


def matgen_pxrdsim(atom,GrainSize,orientation,thermo_vib,deformation,L,H,S,
                   background_order,background_ratio, mixture_noise_ratio,
                   lattice_extinction_ratio,lattice_torsion_ratio,xrd):
    wavelength = 1.54184
    two_theta_range = (10, 80.0,0.02) 

    mu_array,_Ints = get_diff(atom,deformation,lattice_extinction_ratio,lattice_torsion_ratio) 
    
    Γ = 0.888*wavelength/(GrainSize*np.cos(np.radians(np.array(mu_array)/2)))
    gamma_list = Γ / 2 + 1e-10
    sigma2_list = Γ**2 / (8*np.sqrt(2)) + 1e-10

    Ints = []
    for k in range(len(_Ints)):
        
        Ori_coe = np.clip(np.random.normal(loc=1, scale=0.2), 1-orientation[0], 1+orientation[0])
        M = 8/3 * np.pi**2*thermo_vib**2 * (np.sin(np.radians(mu_array[k]/2)) / wavelength)**2
        Deb_coe = np.exp(-2*M)
        Ints.append(_Ints[k] * Ori_coe * Deb_coe)

    x_sim = np.arange(two_theta_range[0],two_theta_range[1],two_theta_range[2])
    y_sim = 0
    for num in range(len(Ints)):
        #_ = draw_peak_density(x_sim, Ints[num], mu_array[num], gamma_list[num], sigma2_list[num])

        _ = combined_peak(x_sim, Ints[num], mu_array[num], gamma_list[num], sigma2_list[num], L,H,S,two_theta_range[2])
        y_sim += _
    # normalize the profile
    nor_y = y_sim / theta_intensity_area(x_sim,y_sim)
    
    random_polynomial = generate_random_polynomial(degree=background_order)
    _bac = random_polynomial(x_sim)
    _bac -= _bac.min()
    _bacI = _bac / _bac.max() * nor_y.max() *background_ratio
    mixture = np.random.uniform(0, nor_y.max() * mixture_noise_ratio, size=len(x_sim))
    nor_y +=  np.flip(_bacI) + mixture
    nor_y = scale_list(nor_y)
    if xrd=='real':
        x_sim = wavelength/(2*np.sin(x_sim/2  * np.pi/ 180 ))
    return x_sim, nor_y