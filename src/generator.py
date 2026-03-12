################################################################
# main.py
################################################################

from ase.io import read
import numpy as np
from ase.db import connect
from ase.spacegroup import get_spacegroup, Spacegroup
from ase import Atoms
from .utils.funs import *
from .utils.MatgenKit import *
from .utils.WPEMsim import *
try:
    from .utils.WPEMsim_torch import pxrdsim_torch
    from .utils.funs_torch import init_precision, get_device, DTYPE
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
import pandas as pd
import json
import copy
import warnings


def parser(database,entry_id,grainsize=20,prefect_orientation=[0.1,0.1],thermo_vibration=0.1,
          zero_shift=0.1,dis_detector2sample=500,half_height_slit_detector = 5,half_height_sample=2.5,
          deformation=False,sim_model=None,xrd='reciprocal', background_order = 6,
          background_ratio=0.05, mixture_noise_ratio=0.02, lattice_extinction_ratio=0.01,lattice_torsion_ratio=0.01,):
    """
    Simulate X-ray diffraction patterns based on a given database file and data ID.

    Parameters:
        db_file (str): Path to the database file (e.g., 'cif.db').
        data_id (int): The ID of the data entry to be processed.

    Optional Parameters:
        deformation (bool, optional): Whether to apply deformation to the lattice. Defaults to False.
        sim_model (str, optional): The simulation model to use. Can be 'WPEM' for WPEM simulation or None for conventional simulation. Defaults to None.
        xrd (str, optional): The type of X-ray diffraction to simulate. Can be 'reciprocal' or 'real'. Defaults to 'reciprocal'.
        
        Sample Parameters:
        grainsize (float, optional): Grain size of the specimen in Angstroms. Defaults to 20.0.
        perfect_orientation (list of float, optional): Perfect orientation of the specimen in degrees. Defaults to [0.1, 0.1].
        lattice_extinction_ratio (float, optional): Ratio of lattice extinction in deformation. Defaults to 0.01.
        lattice_torsion_ratio (float, optional): Ratio of lattice torsion in deformation. Defaults to 0.01.
        
        Testing Condition Parameters:
        thermo_vibration (float, optional): Thermodynamic vibration, the average offset of atoms, in Angstroms. Defaults to 0.1.
        background_order (int, optional): The order of the background. Can be 4 or 6. Defaults to 6.
        background_ratio (float, optional): Ratio of scattering background intensity to peak intensity. Defaults to 0.05.
        mixture_noise_ratio (float, optional): Ratio of mixture vibration noise to peak intensity. Defaults to 0.02.
        
        Instrument Parameters:
        dis_detector2sample (int, optional): Distance between the detector and the sample in mm. Defaults to 500.
        half_height_slit_detector (int, optional): Half height of the slit-shaped detector in mm. Defaults to 5 (2H = 10 mm).
        half_height_sample  (int, optional): Half height of the sample in mm. Defaults to 2.5 (height = 5 mm).
        zero_shift (float, optional): Zero shift of angular position in degrees. Defaults to 0.1.

    Returns:
        tuple: A tuple containing the following elements:
            - x: Lattice plane distance in the x-direction (in Angstroms) if xrd='real', or diffraction angle in the x-direction (in degrees) if xrd='reciprocal'.
            - y: Corresponding diffraction intensity in the y-direction (arbitrary units).

    Example:
        from Pysimxrd import generator
        from ase.db import connect

        database = connect('demo.db')
        entry_id = 1

        x, y = generator.parser(database, entry_id)

        import matplotlib.pyplot as plt
        plt.plot(x,y)
    """

    atoms = database.get_atoms(id=entry_id)
    dis_detector2sample = dis_detector2sample / 10
    if sim_model is None:
        x,y = matgen_pxrdsim(
            atoms,grainsize,prefect_orientation,thermo_vibration,deformation,
            dis_detector2sample,half_height_slit_detector,half_height_sample,background_order,
            background_ratio, mixture_noise_ratio, lattice_extinction_ratio,lattice_torsion_ratio,xrd
            )
    elif sim_model == 'WPEM':
        # for general db files contains conventional lattice cells
        # G_latt_consts = atoms.cell.cellpar()
        # in case the input is primitive unit cell 
        G_latt_consts,_, c_atom = prim2conv(atoms,deformation,lattice_extinction_ratio,lattice_torsion_ratio,)
        G_spacegroup = get_spacegroup(c_atom).no 
        N_symbols = c_atom.get_chemical_symbols() 
        positions = c_atom.get_scaled_positions() 
        G_latt_vol = c_atom.get_volume() 
        spacegroup_obj = get_spacegroup(c_atom)
        spacegroup_symbol = spacegroup_obj.symbol[0]
        crystal_system = space_group_to_crystal_system(G_spacegroup)
        AtomCoordinates = c_atom_covert2WPEMformat(positions, N_symbols)
        x,y = pxrdsim(
                        G_latt_vol, spacegroup_symbol, AtomCoordinates, G_latt_consts, crystal_system,
                        grainsize, prefect_orientation, thermo_vibration, zero_shift,
                        dis_detector2sample,half_height_slit_detector,half_height_sample,
                        background_order,background_ratio, mixture_noise_ratio,  xrd
                        )
    elif sim_model == 'WPEM_torch':
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for sim_model='WPEM_torch'. "
                "Install with: pip install torch")
        G_latt_consts,_, c_atom = prim2conv(atoms,deformation,lattice_extinction_ratio,lattice_torsion_ratio,)
        G_spacegroup = get_spacegroup(c_atom).no
        N_symbols = c_atom.get_chemical_symbols()
        positions = c_atom.get_scaled_positions()
        G_latt_vol = c_atom.get_volume()
        spacegroup_obj = get_spacegroup(c_atom)
        spacegroup_symbol = spacegroup_obj.symbol[0]
        crystal_system = space_group_to_crystal_system(G_spacegroup)
        AtomCoordinates = c_atom_covert2WPEMformat(positions, N_symbols)
        x,y = pxrdsim_torch(
                        G_latt_vol, spacegroup_symbol, AtomCoordinates, G_latt_consts, crystal_system,
                        grainsize, prefect_orientation, thermo_vibration, zero_shift,
                        dis_detector2sample,half_height_slit_detector,half_height_sample,
                        background_order,background_ratio, mixture_noise_ratio,  xrd
                        )
    return x,y




