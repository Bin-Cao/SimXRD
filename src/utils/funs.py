################################################################
# funs.py
################################################################
import json
import os
import numpy as np
import pandas as pd
from sympy import symbols, cos, sin
import sys
import re
from scipy.special import wofz
from ase import Atoms
import spglib

def symbol_to_atomic_number(symbol_list):
    # Mapping of element symbols to atomic numbers
    atomic_numbers = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5,
        'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
        'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
        'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
        'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35,
        'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
        'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45,
        'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55,
        'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
        'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65,
        'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
        'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75,
        'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
        'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
        'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
        'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95,
        'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
        'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
        'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
        'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115,
        'Lv': 116, 'Ts': 117, 'Og': 118
    }
    
    atomic_number_list = []
    for symbol in symbol_list:
        if symbol in atomic_numbers:
            atomic_number_list.append(atomic_numbers[symbol])
        else:
            atomic_number_list.append(0)  # Append None if symbol not in the dictionary
    
    return atomic_number_list


def find_cif_files(directory):
    """
    Walk through the given directory and return a list of paths to .cif files.
    """
    cif_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.cif'):
                cif_files.append(os.path.join(root, file))
    return cif_files


def get_all_distances(positions):
    """
    input:
    positions: numpy array of shape (n, 3) containing the positions of n points in 3D space
    
    output:
    distances: numpy array of shape (n, n) containing the distances between each pair of points
    """
    n = positions.shape[0]
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            distances[i][j] = np.linalg.norm(positions[i] - positions[j])
            distances[j][i] = distances[i][j]  # distances matrix is symmetric
    return distances

def Diffraction_index(system,latt,cal_wavelength,two_theta_range):
    """
    Calculation of Diffraction Peak Positions by Diffraction Geometry
    (S-S') = G*, where G* is the reciprocal lattice vector
    """

    grid = grid_atom()
    index_of_origin = np.where((grid[:, 0] == 0) & (grid[:, 1] == 0) & (grid[:, 2] == 0))[0][0]
    grid[[0, index_of_origin]] = grid[[index_of_origin, 0]]
    d_f = BraggLawDerivation().d_spcing(system)
    sym_h, sym_k, sym_l, sym_a, sym_b, sym_c, angle1, angle2, angle3 = \
            symbols('sym_h sym_k sym_l sym_a sym_b sym_c angle1 angle2 angle3')
    d_list = [1e-10] # HKL=000

    for i in range(len(grid)-1):
        peak = grid[i+1]
        d_list.append(
            float(d_f.subs({sym_h: peak[0], sym_k: peak[1], sym_l: peak[2], sym_a: latt[0], sym_b: latt[1],
                            sym_c: latt[2], angle1: latt[3]*np.pi/180, angle2: latt[4]*np.pi/180, angle3:latt[5]*np.pi/180}))
                            )
        
    # Satisfied the Bragg Law
    # 2theta = 2 * arcsin (lamda / 2 / d)
    bragg_d = cal_wavelength /2/np.array(d_list)
    index0 = np.where(bragg_d > 1)
    # avoid null values
    _d_list = pd.DataFrame(d_list).drop(index0[0])
    _grid = pd.DataFrame(grid).drop(index0[0])

    # recover the index of DataFrame
    for i in [_d_list, _grid]:
        i.index = range(i.shape[0])

    two_theta = 2 * np.arcsin(cal_wavelength /2/np.array(_d_list.iloc[:,0])) * (180 / np.pi)
    index = np.where((two_theta <= two_theta_range[0]) | (two_theta >= two_theta_range[1]))

    d_list = _d_list.drop(index[0])
    grid = _grid.drop(index[0])

    # return all HKL which are satisfied Bragg law
    res_HKL, res_d = de_redundant(grid, d_list)
    return res_HKL, res_d

def de_redundant(grid, d_list):
    """
    Multiplicity due to spatial symmetry
    """
    # input is DataFrame
    grid = np.array(grid.iloc[:,[0,1,2]])
    d_list = np.array(d_list.iloc[:,0])

    res_HKL = []
    res_d = []

    index = -1
    for i in d_list:
        item = get_float(i,4)
        index += 1
        if item not in res_d:
            res_d.append(item)
            res_HKL.append(grid[index])
    return res_HKL, res_d


def get_float(f_str, n):
    f_str = str(f_str)      
    a, _, c = f_str.partition('.')
    c = (c+"0"*n)[:n]       
    return float(".".join([a, c]))

def grid_atom():
    hh, kk, ll = np.mgrid[-11:13:1, -11:13:1, -11:13:1]
    grid = np.c_[hh.ravel(), kk.ravel(), ll.ravel()]
    sorted_indices = np.argsort(np.linalg.norm(grid, axis=1))
    grid = grid[sorted_indices]
    index_of_origin = np.where((grid[:, 0] == 0) & (grid[:, 1] == 0) & (grid[:, 2] == 0))[0][0]
    grid[[0, index_of_origin]] = grid[[index_of_origin, 0]]
    return grid

def cal_extinction(Point_group,HKL_list,dis_list,system,AtomCoordinates,wavelength,cal_extinction=True):
    if cal_extinction == False:
        return HKL_list,[],dis_list,[]
    else:
        HKL_list = np.array(HKL_list).tolist()
        # Diffraction crystal plat
        res_HKL = []
        # interplanar spacing
        d_res_HKL = []

        # extinction crystal plat
        ex_HKL = []
        # interplanar spacing
        d_ex_HKL = []
        for angle in range(len(HKL_list)):
            two_theta = 2 * np.arcsin(wavelength /2/dis_list[angle]) * 180 / np.pi
            l_extinction = lattice_extinction(Point_group,HKL_list[angle],system)
            if l_extinction == True:
                ex_HKL.append(HKL_list[angle])
                d_ex_HKL.append(dis_list[angle])
            else:
                s_extinction = structure_extinction(AtomCoordinates,HKL_list[angle],two_theta,wavelength)
                if s_extinction == True:
                    ex_HKL.append(HKL_list[angle])
                    d_ex_HKL.append(dis_list[angle])
                else:
                    res_HKL.append(HKL_list[angle])
                    d_res_HKL.append(dis_list[angle])
        return res_HKL, ex_HKL, d_res_HKL, d_ex_HKL
    
def cal_extinction(Point_group,HKL_list,dis_list,system,AtomCoordinates,wavelength,cal_extinction=True):
    if cal_extinction == False:
        return HKL_list,[],dis_list,[]
    else:
        HKL_list = np.array(HKL_list).tolist()
        # Diffraction crystal plat
        res_HKL = []
        # interplanar spacing
        d_res_HKL = []

        # extinction crystal plat
        ex_HKL = []
        # interplanar spacing
        d_ex_HKL = []
        for angle in range(len(HKL_list)):
            two_theta = 2 * np.arcsin(wavelength /2/dis_list[angle]) * 180 / np.pi
            l_extinction = lattice_extinction(Point_group,HKL_list[angle],system)
            if l_extinction == True:
                ex_HKL.append(HKL_list[angle])
                d_ex_HKL.append(dis_list[angle])
            else:
                s_extinction = structure_extinction(AtomCoordinates,HKL_list[angle],two_theta,wavelength)
                if s_extinction == True:
                    ex_HKL.append(HKL_list[angle])
                    d_ex_HKL.append(dis_list[angle])
                else:
                    res_HKL.append(HKL_list[angle])
                    d_res_HKL.append(dis_list[angle])
        return res_HKL, ex_HKL, d_res_HKL, d_ex_HKL

def lattice_extinction(lattice_type,HKL,system):
    extinction = False
    # symmetry structures
    if lattice_type == 'P' or lattice_type == 'R':
        pass
    elif lattice_type == 'I': # body center
        if abs((HKL[0]+HKL[1]+HKL[2])) % 2 == 1:
            extinction = True
        else: pass
    elif lattice_type == 'C': # bottom center
        if system == 1 or system == 5: 
            if abs((HKL[0]+HKL[1])) % 2 == 1 or abs((HKL[1]+HKL[2])) % 2 == 1 or abs((HKL[0]+HKL[2])) % 2 == 1: 
                extinction = True
            else: pass
        else:
            if abs((HKL[0]+HKL[1])) % 2 == 1: 
                extinction = True
            else: pass
    elif lattice_type == 'F': # face center
        if (abs(HKL[0]) % 2 == 1 and abs(HKL[1]) % 2 == 1 and abs(HKL[2]) % 2 == 1) or (abs(HKL[0]) % 2 == 0 and abs(HKL[1]) % 2 == 0 and abs(HKL[2]) % 2 == 0):
            pass
        else: extinction = True
    return extinction

def structure_extinction(AtomCoordinates,HKL,two_theta,wavelength):
    # AtomCoordinates = [['Cu2+',0.5,0.5,0.5],[],..]
    extinction = False

    FHKL_square_left = 0
    FHKL_square_right = 0
    for atom in range(len(AtomCoordinates)):
        fi = cal_atoms(AtomCoordinates[atom][0],two_theta, wavelength)
        FHKL_square_left += fi * np.cos(2 * np.pi * (AtomCoordinates[atom][1] * HKL[0] +
                                            AtomCoordinates[atom][2] * HKL[1] + AtomCoordinates[atom][3] * HKL[2]))
        FHKL_square_right += fi * np.sin(2 * np.pi * (AtomCoordinates[atom][1] * HKL[0] +
                                            AtomCoordinates[atom][2] * HKL[1] + AtomCoordinates[atom][3] * HKL[2]))
    FHKL_square = (FHKL_square_left ** 2 + FHKL_square_right ** 2)

    if FHKL_square <= 1e-5:
        extinction = True
    else: pass
    return extinction

# functions defined in Simulatiuon module
def cal_atoms(ion, angle, wavelength,):
    """
    ion : atomic type, i.e., 'Cu2+' 
    angle : 2theta
    returns : form factor at diffraction angle 
    """
    dict =  atomics()
    # in case errors 
    loc = np.sin(angle / 2 * np.pi/180) / wavelength 
    floor_ = get_float(loc,1)
    roof_ = get_float((floor_+ 0.1),1)
    if floor_ == 0.0:
        floor_ = 0
    down_key = '{}'.format(floor_)
    up_key = '{}'.format(roof_)

    down = dict[ion][down_key]
    up = dict[ion][up_key]
    # linear interpolation
    # interval = 0.1 defined in form factor table
    fi = (loc - floor_) / 0.1 * (up-down) + down 
    return fi

def getHeavyatom(s):
    """
    Some atomic ionization forms not defined in the table are replaced by their unionized forms
    """
    # Define a function called getHeavyatom that takes one parameter: s, a string that contains letters and/or non-letter characters.
    return re.sub(r'[^A-Za-z]+', "", s)
    # Use the re.sub() function to replace all non-letter characters in s with an empty string. Return the modified string.


def mult_rule(H, K,L,system):
    """
    Define the multiplicity factor resulting from crystal symmetry
    """
    if system == 1: # Cubic
        if (H == K == 0 and L != 0) or (H == L == 0 and K != 0 ) or  (K == L == 0 and H != 0):
            mult = 6
        elif (H == K !=  0 and L == 0) or (H == L != 0 and K == 0) or (K == L != 0 and H == 0):
            mult = 12
        elif H != K and K != L and H != L and H!=0 and K!=0 and L!=0 :
            mult = 48
        elif H == K == L != 0:
            mult = 8
        else:
            mult = 24
            
    elif system == 2 : # Hexagonal
        if (H == L == 0 and K != 0 ) or  (K == L == 0 and H != 0):
            mult = 6
        elif H == K == 0 and L != 0:
            mult = 2
        elif H == K !=  0 and L == 0:
            mult = 6
        elif H != K and K != L and H != L and H!=0 and K!=0 and L!=0 :
            mult = 24
        elif H == K == L != 0:
            mult = 1
        else:
            mult = 12
    
    elif system == 5: # Trigonal
        if (H == K == 0 and L != 0) or (H == L == 0 and K != 0 ) or  (K == L == 0 and H != 0):
            mult = 6
        elif (H == K !=  0 and L == 0) or (H == L != 0 and K == 0) or (K == L != 0 and H == 0):
            mult = 6
        elif H != K and K != L and H != L and H!=0 and K!=0 and L!=0  :
            mult = 24
        elif H == K == L != 0:
            mult = 1
        else:
            mult = 12

    elif system == 3: # Tetragonal
        if (H == L == 0 and K != 0 ) or  (K == L == 0 and H != 0):
            mult = 4
        elif H == K == 0 and L != 0:
            mult = 2
        elif H == K !=  0 and L == 0:
            mult = 4
        elif  H != K and K != L and H != L and H!=0 and K!=0 and L!=0 :
            mult = 16
        elif H == K == L != 0:
            mult = 1
        else:
            mult = 8

    elif system == 4: # Orthorhombic
        if (H == K == 0 and L != 0) or (H == L == 0 and K != 0 ) or  (K == L == 0 and H != 0):
            mult = 2
        elif H != K and K != L and H != L and H!=0 and K!=0 and L!=0 :
            mult = 8
        elif (H == K == L != 0) or (H == K != 0 and L == 0) or (H == K != L and H != 0 and L != 0):
            mult = 1
        else:
            mult = 4
        
    elif system == 6: # Monoclinic
        if (H == K == 0 and L != 0) or (H == L == 0 and K != 0 ) or  (K == L == 0 and H != 0):
            mult = 2
        elif H != K and K != L and H != L and H!=0 and K!=0 and L!=0 :
            mult = 4
        elif (H == K == L != 0) or (H == K != 0 and L == 0) or (H == K != L and H != 0 and L != 0):
            mult = 1
        elif H != L and H!=0 and L!=0 and K==0:
            mult = 2
        else:
            mult = 4
       
    elif system == 7: # Triclinic
        if (H == K == L != 0) or (H == K != 0 and L == 0) or (H == K != L and H != 0 and L != 0):
            mult = 1
        else:
            mult = 2
    return mult

def draw_peak_density(x, Weight, mu, gamma, sigma2):
    z = ((x-mu) + 1j * gamma) / (np.sqrt(sigma2) * np.sqrt(2))
 
    Voigt = np.real(wofz(z) / (np.sqrt(sigma2) * np.sqrt(2 * np.pi)))
    peak_density = Weight * Voigt
    return peak_density

def lorenz_density(x, mu, gamma):
    """
    :param x: sample data (2theta)
    :param mu: mean (μi)
    :param gamma: FWHM of Lorenz distribution
    :return: Return the probability density of Lorenz distribution
    """
    density = (1 / np.pi) * (gamma / ((x - mu) ** 2 + gamma ** 2))
    return density


def theta_intensity_area(theta_data, intensity):
    n = len(theta_data) - 1
    __area = 0
    for i in range(n):
        __h = (intensity[i] + intensity[i + 1]) / 2
        __l = theta_data[i + 1] - theta_data[i]
        __area += __h * __l
    return __area

def scale_list(lst):
    max_value = max(lst)
    if max_value == 0:
        return [0] * len(lst)
    scaled_list = [x * (100 / max_value) for x in lst]
    return scaled_list

def generate_random_polynomial(degree):
    coefficients = np.random.randn(degree + 1)
    return np.poly1d(coefficients)
"""

def primitive2conveb(atoms):
    conv_lattice, conv_positions, conv_numbers = get_conventional_cell(atoms)
    conventional_atoms = Atoms(cell=conv_lattice, scaled_positions=conv_positions, numbers=conv_numbers, pbc=True)

    return conventional_atoms.cell.cellpar()

def get_conventional_cell(atoms):
    lattice = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    cell = (lattice, positions, numbers)
    conventional_cell = spglib.standardize_cell(cell, to_primitive=False, no_idealize=True)
    return conventional_cell
"""
def conlattcell2asu(atoms,positions,N_symbols):
    cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    dataset = spglib.get_symmetry_dataset(cell, symprec=1e-3)
    wyckoff_positions = dataset['equivalent_atoms']
    unique_indices = list(set(wyckoff_positions))
    unique_positions = [positions[i] for i in unique_indices]
    unique_symbols = [N_symbols[j] for j in unique_indices]


    equivalent_atoms = {}
    for i, wyckoff_pos in enumerate(wyckoff_positions):
        if wyckoff_pos not in equivalent_atoms:
            equivalent_atoms[wyckoff_pos] = []
        equivalent_atoms[wyckoff_pos].append(i)
    """
    for wyckoff_pos, atom_list in equivalent_atoms.items():
        print(f"Wyckoff position {wyckoff_pos}: Atoms {atom_list}")
    
    >>>
    Equivalent atoms mapping:
    Wyckoff position 0: Atoms [0, 1, 2, 3, 4, 5, 6, 7]
    Wyckoff position 8: Atoms [8, 9, 10, 11, 12, 13, 14, 15]
    Wyckoff position 16: Atoms [16, 17, 18, 19, 20, 21, 22, 23]
    Wyckoff position 24: Atoms [24, 25, 26, 27, 28, 29, 30, 31]
    """

    kinds = np.zeros(len(positions), dtype=int)
    kinds.tolist()
    for k, (wyckoff_pos, atom_list) in enumerate(equivalent_atoms.items()):
        for index in atom_list:
            kinds[index] = int(k)
    
    return unique_positions,unique_symbols,atoms.get_scaled_positions(), kinds

def prim2conv(prim_atom,deformation = False,lattice_extinction_ratio=0.01,lattice_torsion_ratio=0.01,):
    """
    Convert a primitive cell to a conventional cell.

    Parameters:
        prim_atom (Atoms): The primitive atom defined in the atomic simulation unit (asu).

    Returns:
        tuple: Lattice constants, conventional lattice cell matrix in Cartesian coordinates, Atoms attribute
    """
    lattice = prim_atom.get_cell()
    positions = prim_atom.get_scaled_positions()
    numbers = prim_atom.get_atomic_numbers()
    cell = (lattice, positions, numbers)
    conventional_cell = spglib.standardize_cell(cell, to_primitive=False, no_idealize=True)
    conv_lattice, conv_positions, conv_numbers = conventional_cell

    if deformation:
        # No Rotation for assuring the space group symmetry
        a,b,c = np.random.uniform(1-lattice_extinction_ratio, 1+lattice_extinction_ratio, 3)
        stretching_compression = np.array([
                                        [a, 0.0, 0.0],
                                        [0.0, b, 0.0],
                                        [0.0, 0.0, c]
                                    ])  
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = np.random.uniform(0, lattice_torsion_ratio, 6)
        shear = np.array([
                        [1.0, alpha1, alpha2],
                        [beta1, 1.0, beta2],
                        [gamma1, gamma2, 1.0]
                    ])
        conv_lattice = np.dot(stretching_compression , conv_lattice)
        conv_lattice = np.dot(shear , conv_lattice)
     
    else: pass

    conventional_atoms = Atoms(cell=conv_lattice, scaled_positions=conv_positions, numbers=conv_numbers, pbc=True)
    lc = conventional_atoms.cell.cellpar()
    lmtx = conventional_atoms.get_cell()[:]
    return lc, lmtx, conventional_atoms



def atomics():
    _dict = {'H': {'0': 1, '0.1': 0.81, '0.2': 0.48, '0.3': 0.25, '0.4': 0.13, '0.5': 0.07, '0.6': 0.04, '0.7': 0.03, '0.8': 0.02, '0.9': 0.01, '1': 0.0, '1.1': 0.0, '1.2': 100.0}, 'He': {'0': 2, '0.1': 1.88, '0.2': 1.46, '0.3': 1.05, '0.4': 0.75, '0.5': 0.52, '0.6': 0.35, '0.7': 0.24, '0.8': 0.18, '0.9': 0.14, '1': 0.11, '1.1': 0.09, '1.2': 100.0}, 'Li+': {'0': 2, '0.1': 1.96, '0.2': 1.8, '0.3': 1.5, '0.4': 1.3, '0.5': 1.0, '0.6': 0.8, '0.7': 0.6, '0.8': 0.5, '0.9': 0.4, '1': 0.3, '1.1': 0.3, '1.2': 100.0}, 'Li': {'0': 3, '0.1': 2.2, '0.2': 1.8, '0.3': 1.5, '0.4': 1.3, '0.5': 1.0, '0.6': 0.8, '0.7': 0.6, '0.8': 0.5, '0.9': 0.4, '1': 0.3, '1.1': 0.3, '1.2': 100.0}, 'Be2+': {'0': 2, '0.1': 2.0, '0.2': 1.9, '0.3': 1.7, '0.4': 1.6, '0.5': 1.4, '0.6': 1.2, '0.7': 1.0, '0.8': 0.9, '0.9': 0.7, '1': 0.6, '1.1': 0.5, '1.2': 100.0}, 'Be': {'0': 4, '0.1': 2.9, '0.2': 1.9, '0.3': 1.7, '0.4': 1.6, '0.5': 1.4, '0.6': 1.2, '0.7': 1.0, '0.8': 0.9, '0.9': 0.7, '1': 0.6, '1.1': 0.5, '1.2': 100.0}, 'B3+': {'0': 2, '0.1': 1.99, '0.2': 1.9, '0.3': 1.8, '0.4': 1.7, '0.5': 1.6, '0.6': 1.4, '0.7': 1.3, '0.8': 1.2, '0.9': 1.0, '1': 0.9, '1.1': 0.7, '1.2': 100.0}, 'B': {'0': 5, '0.1': 3.5, '0.2': 2.4, '0.3': 1.9, '0.4': 1.7, '0.5': 1.5, '0.6': 1.4, '0.7': 1.2, '0.8': 1.2, '0.9': 1.0, '1': 0.9, '1.1': 0.7, '1.2': 100.0}, 'C': {'0': 6, '0.1': 4.6, '0.2': 3.0, '0.3': 2.2, '0.4': 1.9, '0.5': 1.7, '0.6': 1.6, '0.7': 1.4, '0.8': 1.3, '0.9': 1.16, '1': 1.0, '1.1': 0.9, '1.2': 100.0}, 'N5+': {'0': 2, '0.1': 2.0, '0.2': 2.0, '0.3': 1.9, '0.4': 1.9, '0.5': 1.8, '0.6': 1.7, '0.7': 1.6, '0.8': 1.5, '0.9': 1.4, '1': 1.3, '1.1': 1.16, '1.2': 100.0}, 'N3+': {'0': 4, '0.1': 3.7, '0.2': 3.0, '0.3': 2.4, '0.4': 2.0, '0.5': 1.8, '0.6': 1.66, '0.7': 1.56, '0.8': 1.49, '0.9': 1.39, '1': 1.28, '1.1': 1.17, '1.2': 100.0}, 'N': {'0': 7, '0.1': 5.8, '0.2': 4.2, '0.3': 3.0, '0.4': 2.3, '0.5': 1.9, '0.6': 1.65, '0.7': 1.54, '0.8': 1.49, '0.9': 1.39, '1': 1.29, '1.1': 1.17, '1.2': 100.0}, 'O': {'0': 8, '0.1': 7.1, '0.2': 5.3, '0.3': 3.9, '0.4': 2.9, '0.5': 2.2, '0.6': 1.8, '0.7': 1.6, '0.8': 1.5, '0.9': 1.4, '1': 1.35, '1.1': 1.26, '1.2': 100.0}, 'O2-': {'0': 10, '0.1': 8.0, '0.2': 5.5, '0.3': 3.8, '0.4': 2.7, '0.5': 2.1, '0.6': 1.8, '0.7': 1.5, '0.8': 1.5, '0.9': 1.4, '1': 1.35, '1.1': 1.26, '1.2': 100.0}, 'F': {'0': 9, '0.1': 7.8, '0.2': 6.2, '0.3': 4.45, '0.4': 3.35, '0.5': 2.65, '0.6': 2.15, '0.7': 1.9, '0.8': 1.7, '0.9': 1.6, '1': 1.5, '1.1': 1.35, '1.2': 100.0}, 'F-': {'0': 10, '0.1': 8.7, '0.2': 6.7, '0.3': 4.8, '0.4': 3.5, '0.5': 2.8, '0.6': 2.2, '0.7': 1.9, '0.8': 1.7, '0.9': 1.55, '1': 1.5, '1.1': 1.35, '1.2': 100.0}, 'Ne': {'0': 10, '0.1': 9.3, '0.2': 7.5, '0.3': 5.8, '0.4': 4.4, '0.5': 3.4, '0.6': 2.65, '0.7': 2.2, '0.8': 1.9, '0.9': 1.65, '1': 1.55, '1.1': 1.5, '1.2': 100.0}, 'Na+': {'0': 10, '0.1': 9.5, '0.2': 8.2, '0.3': 6.7, '0.4': 5.25, '0.5': 4.05, '0.6': 3.2, '0.7': 2.65, '0.8': 2.25, '0.9': 1.95, '1': 1.75, '1.1': 1.6, '1.2': 100.0}, 'Na': {'0': 11, '0.1': 9.65, '0.2': 8.2, '0.3': 6.7, '0.4': 5.25, '0.5': 4.05, '0.6': 3.2, '0.7': 2.65, '0.8': 2.25, '0.9': 1.95, '1': 1.75, '1.1': 1.6, '1.2': 100.0}, 'Mg2+': {'0': 10, '0.1': 9.75, '0.2': 8.6, '0.3': 7.25, '0.4': 5.95, '0.5': 4.8, '0.6': 3.85, '0.7': 3.15, '0.8': 2.55, '0.9': 2.2, '1': 2.0, '1.1': 1.8, '1.2': 100.0}, 'Mg': {'0': 12, '0.1': 10.5, '0.2': 8.6, '0.3': 7.25, '0.4': 5.95, '0.5': 4.8, '0.6': 3.85, '0.7': 3.15, '0.8': 2.55, '0.9': 2.2, '1': 2.0, '1.1': 1.8, '1.2': 100.0}, 'Al3+': {'0': 10, '0.1': 9.7, '0.2': 8.9, '0.3': 7.8, '0.4': 6.65, '0.5': 5.5, '0.6': 4.45, '0.7': 3.65, '0.8': 3.1, '0.9': 2.65, '1': 2.3, '1.1': 2.0, '1.2': 100.0}, 'Al': {'0': 13, '0.1': 11.0, '0.2': 8.95, '0.3': 7.75, '0.4': 6.6, '0.5': 5.5, '0.6': 4.5, '0.7': 3.7, '0.8': 3.1, '0.9': 2.65, '1': 2.3, '1.1': 2.0, '1.2': 100.0}, 'Si4+': {'0': 10, '0.1': 9.75, '0.2': 9.15, '0.3': 8.25, '0.4': 7.15, '0.5': 6.05, '0.6': 5.05, '0.7': 4.2, '0.8': 3.4, '0.9': 2.95, '1': 2.6, '1.1': 2.3, '1.2': 100.0}, 'Si': {'0': 14, '0.1': 11.35, '0.2': 9.4, '0.3': 8.2, '0.4': 7.15, '0.5': 6.1, '0.6': 5.1, '0.7': 4.2, '0.8': 3.4, '0.9': 2.95, '1': 2.6, '1.1': 2.3, '1.2': 100.0}, 'P5+': {'0': 10, '0.1': 9.8, '0.2': 9.25, '0.3': 8.45, '0.4': 7.5, '0.5': 6.55, '0.6': 5.65, '0.7': 4.8, '0.8': 4.05, '0.9': 3.4, '1': 3.0, '1.1': 2.6, '1.2': 100.0}, 'P': {'0': 15, '0.1': 12.4, '0.2': 10.0, '0.3': 8.45, '0.4': 7.45, '0.5': 6.5, '0.6': 5.65, '0.7': 4.8, '0.8': 4.05, '0.9': 3.4, '1': 3.0, '1.1': 2.6, '1.2': 100.0}, 'P3-': {'0': 18, '0.1': 12.7, '0.2': 9.8, '0.3': 8.4, '0.4': 7.45, '0.5': 6.5, '0.6': 5.65, '0.7': 4.85, '0.8': 4.05, '0.9': 3.4, '1': 3.0, '1.1': 2.6, '1.2': 100.0}, 'S6+': {'0': 10, '0.1': 9.85, '0.2': 9.4, '0.3': 8.7, '0.4': 7.85, '0.5': 6.85, '0.6': 6.05, '0.7': 5.25, '0.8': 4.5, '0.9': 3.9, '1': 3.35, '1.1': 2.9, '1.2': 100.0}, 'S': {'0': 16, '0.1': 13.6, '0.2': 10.7, '0.3': 8.95, '0.4': 7.85, '0.5': 6.85, '0.6': 6.0, '0.7': 5.25, '0.8': 4.5, '0.9': 3.9, '1': 3.35, '1.1': 2.9, '1.2': 100.0}, 'S2-': {'0': 18, '0.1': 14.3, '0.2': 10.7, '0.3': 8.9, '0.4': 7.85, '0.5': 6.85, '0.6': 6.0, '0.7': 5.25, '0.8': 4.5, '0.9': 3.9, '1': 3.35, '1.1': 2.9, '1.2': 100.0}, 'Cl': {'0': 17, '0.1': 14.6, '0.2': 11.3, '0.3': 9.25, '0.4': 8.05, '0.5': 7.25, '0.6': 6.5, '0.7': 5.75, '0.8': 5.05, '0.9': 4.4, '1': 3.85, '1.1': 3.35, '1.2': 100.0}, 'Cl-': {'0': 18, '0.1': 15.2, '0.2': 11.5, '0.3': 9.3, '0.4': 8.05, '0.5': 7.25, '0.6': 6.5, '0.7': 5.75, '0.8': 5.05, '0.9': 4.4, '1': 3.85, '1.1': 3.35, '1.2': 100.0}, 'A': {'0': 18, '0.1': 15.9, '0.2': 12.6, '0.3': 10.4, '0.4': 8.7, '0.5': 7.8, '0.6': 7.0, '0.7': 6.2, '0.8': 5.4, '0.9': 4.7, '1': 4.1, '1.1': 3.6, '1.2': 100.0}, 'K+': {'0': 18, '0.1': 16.5, '0.2': 13.3, '0.3': 10.8, '0.4': 8.85, '0.5': 7.75, '0.6': 7.05, '0.7': 6.44, '0.8': 5.9, '0.9': 5.3, '1': 4.8, '1.1': 4.2, '1.2': 100.0}, 'Ca2+': {'0': 18, '0.1': 16.8, '0.2': 14.0, '0.3': 11.5, '0.4': 9.3, '0.5': 8.1, '0.6': 7.35, '0.7': 6.7, '0.8': 6.2, '0.9': 5.7, '1': 5.1, '1.1': 4.6, '1.2': 100.0}, 'Sc3+': {'0': 18, '0.1': 16.7, '0.2': 14.0, '0.3': 11.4, '0.4': 9.4, '0.5': 8.3, '0.6': 7.6, '0.7': 6.9, '0.8': 6.4, '0.9': 5.8, '1': 5.35, '1.1': 4.85, '1.2': 100.0}, 'Ti4+': {'0': 18, '0.1': 17.0, '0.2': 14.4, '0.3': 11.9, '0.4': 9.9, '0.5': 8.5, '0.6': 7.85, '0.7': 7.3, '0.8': 6.7, '0.9': 6.15, '1': 5.65, '1.1': 5.05, '1.2': 100.0}, 'Rb+': {'0': 36, '0.1': 33.6, '0.2': 28.7, '0.3': 24.6, '0.4': 21.4, '0.5': 18.9, '0.6': 16.7, '0.7': 14.6, '0.8': 12.8, '0.9': 11.2, '1': 9.9, '1.1': 8.9, '1.2': 100.0}, 'K': {'0': 19, '0.1': 16.5, '0.2': 13.3, '0.3': 10.8, '0.4': 9.2, '0.5': 7.9, '0.6': 6.7, '0.7': 5.9, '0.8': 5.2, '0.9': 4.6, '1': 4.2, '1.1': 3.7, '1.2': 3.3}, 'Ca': {'0': 20, '0.1': 17.5, '0.2': 14.1, '0.3': 11.4, '0.4': 9.7, '0.5': 8.4, '0.6': 7.3, '0.7': 6.3, '0.8': 5.6, '0.9': 4.9, '1': 4.5, '1.1': 4.0, '1.2': 3.6}, 'Sc': {'0': 21, '0.1': 18.4, '0.2': 14.9, '0.3': 12.1, '0.4': 10.3, '0.5': 8.9, '0.6': 7.7, '0.7': 6.7, '0.8': 5.9, '0.9': 5.3, '1': 4.7, '1.1': 4.3, '1.2': 3.9}, 'Ti': {'0': 22, '0.1': 19.3, '0.2': 15.7, '0.3': 12.8, '0.4': 10.9, '0.5': 9.5, '0.6': 8.2, '0.7': 7.2, '0.8': 6.3, '0.9': 5.6, '1': 5.0, '1.1': 4.6, '1.2': 4.2}, 'V': {'0': 23, '0.1': 20.2, '0.2': 16.6, '0.3': 13.5, '0.4': 11.5, '0.5': 10.1, '0.6': 8.7, '0.7': 7.6, '0.8': 6.7, '0.9': 5.9, '1': 5.3, '1.1': 4.9, '1.2': 4.4}, 'Cr': {'0': 24, '0.1': 21.1, '0.2': 17.4, '0.3': 14.2, '0.4': 12.1, '0.5': 10.6, '0.6': 9.2, '0.7': 8.0, '0.8': 7.1, '0.9': 6.3, '1': 5.7, '1.1': 5.1, '1.2': 4.6}, 'Mn': {'0': 25, '0.1': 22.1, '0.2': 18.2, '0.3': 14.9, '0.4': 12.7, '0.5': 11.1, '0.6': 9.7, '0.7': 8.4, '0.8': 7.5, '0.9': 6.6, '1': 6.0, '1.1': 5.4, '1.2': 4.9}, 'Fe': {'0': 26, '0.1': 23.1, '0.2': 18.9, '0.3': 15.6, '0.4': 13.3, '0.5': 11.6, '0.6': 10.2, '0.7': 8.9, '0.8': 7.9, '0.9': 7.0, '1': 6.3, '1.1': 5.7, '1.2': 5.2}, 'Co': {'0': 27, '0.1': 24.1, '0.2': 19.8, '0.3': 16.4, '0.4': 14.0, '0.5': 12.1, '0.6': 10.7, '0.7': 9.3, '0.8': 8.3, '0.9': 7.3, '1': 6.7, '1.1': 6.0, '1.2': 5.5}, 'Ni': {'0': 28, '0.1': 25.0, '0.2': 20.7, '0.3': 17.2, '0.4': 14.6, '0.5': 12.7, '0.6': 11.2, '0.7': 9.8, '0.8': 8.7, '0.9': 7.7, '1': 7.0, '1.1': 6.3, '1.2': 5.8}, 'Cu': {'0': 29, '0.1': 25.9, '0.2': 21.6, '0.3': 17.9, '0.4': 15.2, '0.5': 13.3, '0.6': 11.7, '0.7': 10.2, '0.8': 9.1, '0.9': 8.1, '1': 7.3, '1.1': 6.6, '1.2': 6.0}, 'Zn': {'0': 30, '0.1': 26.8, '0.2': 22.4, '0.3': 18.6, '0.4': 15.8, '0.5': 13.9, '0.6': 12.2, '0.7': 10.7, '0.8': 9.6, '0.9': 8.5, '1': 7.6, '1.1': 6.9, '1.2': 6.3}, 'Ga': {'0': 31, '0.1': 27.8, '0.2': 23.3, '0.3': 19.3, '0.4': 16.5, '0.5': 14.5, '0.6': 12.7, '0.7': 11.2, '0.8': 10.0, '0.9': 8.9, '1': 7.9, '1.1': 7.3, '1.2': 6.7}, 'Ge': {'0': 32, '0.1': 28.8, '0.2': 24.1, '0.3': 20.0, '0.4': 17.1, '0.5': 15.0, '0.6': 13.2, '0.7': 11.6, '0.8': 10.4, '0.9': 9.3, '1': 8.3, '1.1': 7.6, '1.2': 7.0}, 'As': {'0': 33, '0.1': 29.7, '0.2': 25.0, '0.3': 20.8, '0.4': 17.7, '0.5': 15.6, '0.6': 13.8, '0.7': 12.1, '0.8': 10.8, '0.9': 9.7, '1': 8.7, '1.1': 7.9, '1.2': 7.3}, 'Se': {'0': 34, '0.1': 30.6, '0.2': 25.8, '0.3': 21.5, '0.4': 18.3, '0.5': 16.1, '0.6': 14.3, '0.7': 12.6, '0.8': 11.2, '0.9': 10.0, '1': 9.0, '1.1': 8.2, '1.2': 7.5}, 'Br': {'0': 35, '0.1': 31.6, '0.2': 26.6, '0.3': 22.3, '0.4': 18.9, '0.5': 16.7, '0.6': 14.8, '0.7': 13.1, '0.8': 11.7, '0.9': 10.4, '1': 9.4, '1.1': 8.6, '1.2': 7.8}, 'Kr': {'0': 36, '0.1': 32.5, '0.2': 27.4, '0.3': 23.0, '0.4': 19.5, '0.5': 17.3, '0.6': 15.3, '0.7': 13.6, '0.8': 12.1, '0.9': 10.8, '1': 9.8, '1.1': 8.9, '1.2': 8.1}, 'Rb': {'0': 37, '0.1': 33.5, '0.2': 28.2, '0.3': 23.8, '0.4': 20.2, '0.5': 17.9, '0.6': 15.9, '0.7': 14.1, '0.8': 12.5, '0.9': 11.2, '1': 10.2, '1.1': 9.2, '1.2': 8.4}, 'Sr': {'0': 38, '0.1': 34.4, '0.2': 29.0, '0.3': 24.5, '0.4': 20.8, '0.5': 18.4, '0.6': 16.4, '0.7': 14.6, '0.8': 12.9, '0.9': 11.6, '1': 10.5, '1.1': 9.5, '1.2': 8.7}, 'Y': {'0': 39, '0.1': 35.4, '0.2': 29.9, '0.3': 25.3, '0.4': 21.5, '0.5': 19.0, '0.6': 17.0, '0.7': 15.1, '0.8': 13.4, '0.9': 12.0, '1': 10.9, '1.1': 9.9, '1.2': 9.0}, 'Zr': {'0': 40, '0.1': 36.3, '0.2': 30.8, '0.3': 26.0, '0.4': 22.1, '0.5': 19.7, '0.6': 17.5, '0.7': 15.6, '0.8': 13.8, '0.9': 12.4, '1': 11.2, '1.1': 10.2, '1.2': 9.3}, 'Nb': {'0': 41, '0.1': 37.3, '0.2': 31.7, '0.3': 26.8, '0.4': 22.8, '0.5': 20.2, '0.6': 18.1, '0.7': 16.0, '0.8': 14.3, '0.9': 12.8, '1': 11.6, '1.1': 10.6, '1.2': 9.7}, 'Mo': {'0': 42, '0.1': 38.2, '0.2': 32.6, '0.3': 27.6, '0.4': 23.5, '0.5': 20.8, '0.6': 18.6, '0.7': 16.5, '0.8': 14.8, '0.9': 13.2, '1': 12.0, '1.1': 10.9, '1.2': 10.0}, 'Tc': {'0': 43, '0.1': 39.1, '0.2': 33.4, '0.3': 28.3, '0.4': 24.1, '0.5': 21.3, '0.6': 19.1, '0.7': 17.0, '0.8': 15.2, '0.9': 13.6, '1': 12.3, '1.1': 11.3, '1.2': 10.3}, 'Ru': {'0': 44, '0.1': 40.0, '0.2': 34.3, '0.3': 29.1, '0.4': 24.7, '0.5': 21.9, '0.6': 19.6, '0.7': 17.5, '0.8': 15.6, '0.9': 14.1, '1': 12.7, '1.1': 11.6, '1.2': 10.6}, 'Rh': {'0': 45, '0.1': 41.0, '0.2': 35.1, '0.3': 29.9, '0.4': 25.4, '0.5': 22.5, '0.6': 20.2, '0.7': 18.0, '0.8': 16.1, '0.9': 14.5, '1': 13.1, '1.1': 12.0, '1.2': 11.0}, 'Pd': {'0': 46, '0.1': 41.9, '0.2': 36.0, '0.3': 30.7, '0.4': 26.2, '0.5': 23.1, '0.6': 20.8, '0.7': 18.5, '0.8': 16.5, '0.9': 14.9, '1': 13.6, '1.1': 12.3, '1.2': 11.3}, 'Ag': {'0': 47, '0.1': 42.8, '0.2': 36.9, '0.3': 31.5, '0.4': 26.9, '0.5': 23.8, '0.6': 21.3, '0.7': 19.0, '0.8': 17.1, '0.9': 15.3, '1': 14.0, '1.1': 12.7, '1.2': 11.7}, 'Cd': {'0': 48, '0.1': 43.7, '0.2': 37.7, '0.3': 32.2, '0.4': 27.5, '0.5': 24.4, '0.6': 21.8, '0.7': 19.6, '0.8': 17.6, '0.9': 15.7, '1': 14.3, '1.1': 13.0, '1.2': 12.0}, 'In': {'0': 49, '0.1': 44.7, '0.2': 38.6, '0.3': 33.0, '0.4': 28.1, '0.5': 25.0, '0.6': 22.4, '0.7': 20.1, '0.8': 18.0, '0.9': 16.2, '1': 14.7, '1.1': 13.4, '1.2': 12.3}, 'Sn': {'0': 50, '0.1': 45.7, '0.2': 39.5, '0.3': 33.8, '0.4': 28.7, '0.5': 25.6, '0.6': 22.9, '0.7': 20.6, '0.8': 18.5, '0.9': 16.6, '1': 15.1, '1.1': 13.7, '1.2': 12.7}, 'Sb': {'0': 51, '0.1': 46.7, '0.2': 40.4, '0.3': 34.6, '0.4': 29.5, '0.5': 26.3, '0.6': 23.5, '0.7': 21.1, '0.8': 19.0, '0.9': 17.0, '1': 15.5, '1.1': 14.1, '1.2': 13.0}, 'Te': {'0': 52, '0.1': 47.7, '0.2': 41.3, '0.3': 35.4, '0.4': 30.3, '0.5': 26.9, '0.6': 24.0, '0.7': 21.7, '0.8': 19.5, '0.9': 17.5, '1': 16.0, '1.1': 14.5, '1.2': 13.3}, 'I': {'0': 53, '0.1': 48.6, '0.2': 42.1, '0.3': 36.1, '0.4': 31.0, '0.5': 27.5, '0.6': 24.6, '0.7': 22.2, '0.8': 20.0, '0.9': 17.9, '1': 16.4, '1.1': 14.8, '1.2': 13.6}, 'Xe': {'0': 54, '0.1': 49.6, '0.2': 43.0, '0.3': 36.8, '0.4': 31.6, '0.5': 28.0, '0.6': 25.2, '0.7': 22.7, '0.8': 20.4, '0.9': 18.4, '1': 16.7, '1.1': 15.2, '1.2': 13.9}, 'Cs': {'0': 55, '0.1': 50.7, '0.2': 43.8, '0.3': 37.6, '0.4': 32.4, '0.5': 28.7, '0.6': 25.8, '0.7': 23.2, '0.8': 20.8, '0.9': 18.8, '1': 17.0, '1.1': 15.6, '1.2': 14.5}, 'Ba': {'0': 56, '0.1': 51.7, '0.2': 44.7, '0.3': 38.4, '0.4': 33.1, '0.5': 29.3, '0.6': 26.4, '0.7': 23.7, '0.8': 21.3, '0.9': 19.2, '1': 17.4, '1.1': 16.0, '1.2': 14.7}, 'La': {'0': 57, '0.1': 52.6, '0.2': 45.6, '0.3': 39.3, '0.4': 33.8, '0.5': 29.8, '0.6': 26.9, '0.7': 24.3, '0.8': 21.9, '0.9': 19.7, '1': 17.9, '1.1': 16.4, '1.2': 15.0}, 'Ce': {'0': 58, '0.1': 53.6, '0.2': 46.5, '0.3': 40.1, '0.4': 34.5, '0.5': 30.4, '0.6': 27.4, '0.7': 24.8, '0.8': 22.4, '0.9': 20.2, '1': 18.4, '1.1': 16.6, '1.2': 15.3}, 'Pr': {'0': 59, '0.1': 54.5, '0.2': 47.4, '0.3': 40.9, '0.4': 35.2, '0.5': 31.1, '0.6': 28.0, '0.7': 25.4, '0.8': 22.9, '0.9': 20.6, '1': 18.8, '1.1': 17.1, '1.2': 15.7}, 'Nd': {'0': 60, '0.1': 55.4, '0.2': 48.3, '0.3': 41.6, '0.4': 35.9, '0.5': 31.8, '0.6': 28.6, '0.7': 25.9, '0.8': 23.4, '0.9': 21.1, '1': 19.2, '1.1': 17.5, '1.2': 16.1}, 'Pm': {'0': 61, '0.1': 56.4, '0.2': 49.1, '0.3': 42.4, '0.4': 36.6, '0.5': 32.4, '0.6': 29.2, '0.7': 26.4, '0.8': 23.9, '0.9': 21.5, '1': 19.6, '1.1': 17.9, '1.2': 16.4}, 'Sm': {'0': 62, '0.1': 57.3, '0.2': 50.0, '0.3': 43.2, '0.4': 37.3, '0.5': 32.9, '0.6': 29.8, '0.7': 26.9, '0.8': 24.4, '0.9': 22.0, '1': 20.0, '1.1': 18.3, '1.2': 16.8}, 'Eu': {'0': 63, '0.1': 58.3, '0.2': 50.9, '0.3': 44.0, '0.4': 38.1, '0.5': 33.5, '0.6': 30.4, '0.7': 27.5, '0.8': 24.9, '0.9': 22.4, '1': 20.4, '1.1': 18.7, '1.2': 17.1}, 'Gd': {'0': 64, '0.1': 59.3, '0.2': 51.7, '0.3': 44.8, '0.4': 38.8, '0.5': 34.1, '0.6': 31.0, '0.7': 28.1, '0.8': 25.4, '0.9': 22.9, '1': 20.8, '1.1': 19.1, '1.2': 17.5}, 'Tb': {'0': 65, '0.1': 60.2, '0.2': 52.6, '0.3': 45.7, '0.4': 39.6, '0.5': 34.7, '0.6': 31.6, '0.7': 28.6, '0.8': 25.9, '0.9': 23.4, '1': 21.2, '1.1': 19.5, '1.2': 17.9}, 'Dy': {'0': 66, '0.1': 61.1, '0.2': 53.6, '0.3': 46.5, '0.4': 40.4, '0.5': 35.4, '0.6': 32.2, '0.7': 29.2, '0.8': 26.3, '0.9': 23.9, '1': 21.6, '1.1': 19.9, '1.2': 18.3}, 'Ho': {'0': 67, '0.1': 62.1, '0.2': 54.5, '0.3': 47.3, '0.4': 41.1, '0.5': 36.1, '0.6': 32.7, '0.7': 29.7, '0.8': 26.8, '0.9': 24.3, '1': 22.0, '1.1': 20.3, '1.2': 18.6}, 'Er': {'0': 68, '0.1': 63.0, '0.2': 55.3, '0.3': 48.1, '0.4': 41.7, '0.5': 36.7, '0.6': 33.3, '0.7': 30.2, '0.8': 27.3, '0.9': 24.7, '1': 22.4, '1.1': 20.7, '1.2': 18.9}, 'Tu': {'0': 69, '0.1': 64.0, '0.2': 56.2, '0.3': 48.9, '0.4': 42.4, '0.5': 37.4, '0.6': 33.9, '0.7': 30.8, '0.8': 27.9, '0.9': 25.2, '1': 22.9, '1.1': 21.0, '1.2': 19.3}, 'Yb': {'0': 70, '0.1': 64.9, '0.2': 57.0, '0.3': 49.7, '0.4': 43.2, '0.5': 38.0, '0.6': 34.4, '0.7': 31.3, '0.8': 28.4, '0.9': 25.7, '1': 23.3, '1.1': 21.4, '1.2': 19.7}, 'Lu': {'0': 71, '0.1': 65.9, '0.2': 57.8, '0.3': 50.4, '0.4': 43.9, '0.5': 38.7, '0.6': 35.0, '0.7': 31.8, '0.8': 28.9, '0.9': 26.2, '1': 23.8, '1.1': 21.8, '1.2': 20.0}, 'Hf': {'0': 72, '0.1': 66.8, '0.2': 58.6, '0.3': 51.2, '0.4': 44.5, '0.5': 39.3, '0.6': 35.6, '0.7': 32.3, '0.8': 29.3, '0.9': 26.7, '1': 24.2, '1.1': 22.3, '1.2': 20.4}, 'Ta': {'0': 73, '0.1': 67.8, '0.2': 59.5, '0.3': 52.0, '0.4': 45.3, '0.5': 39.9, '0.6': 36.2, '0.7': 32.9, '0.8': 29.8, '0.9': 27.1, '1': 24.7, '1.1': 22.6, '1.2': 20.9}, 'W': {'0': 74, '0.1': 68.8, '0.2': 60.4, '0.3': 52.8, '0.4': 46.1, '0.5': 40.5, '0.6': 36.8, '0.7': 33.5, '0.8': 30.4, '0.9': 27.6, '1': 25.2, '1.1': 23.0, '1.2': 21.3}, 'Re': {'0': 75, '0.1': 69.8, '0.2': 61.3, '0.3': 53.6, '0.4': 46.8, '0.5': 41.1, '0.6': 37.4, '0.7': 34.0, '0.8': 30.9, '0.9': 28.1, '1': 25.6, '1.1': 23.4, '1.2': 21.6}, 'Os': {'0': 76, '0.1': 70.8, '0.2': 62.2, '0.3': 54.4, '0.4': 47.5, '0.5': 41.7, '0.6': 38.0, '0.7': 34.6, '0.8': 31.4, '0.9': 28.6, '1': 26.0, '1.1': 23.9, '1.2': 22.0}, 'Ir': {'0': 77, '0.1': 71.7, '0.2': 63.1, '0.3': 55.3, '0.4': 48.2, '0.5': 42.4, '0.6': 38.6, '0.7': 35.1, '0.8': 32.0, '0.9': 29.0, '1': 26.5, '1.1': 24.3, '1.2': 22.3}, 'Pt': {'0': 78, '0.1': 72.6, '0.2': 64.0, '0.3': 56.2, '0.4': 48.9, '0.5': 43.1, '0.6': 39.2, '0.7': 35.6, '0.8': 32.5, '0.9': 29.5, '1': 27.0, '1.1': 24.7, '1.2': 22.7}, 'Au': {'0': 79, '0.1': 73.6, '0.2': 65.0, '0.3': 57.0, '0.4': 49.7, '0.5': 43.8, '0.6': 39.8, '0.7': 36.2, '0.8': 33.1, '0.9': 30.0, '1': 27.4, '1.1': 25.1, '1.2': 23.1}, 'Hg': {'0': 80, '0.1': 74.6, '0.2': 65.9, '0.3': 57.9, '0.4': 50.5, '0.5': 44.4, '0.6': 40.5, '0.7': 36.8, '0.8': 33.6, '0.9': 30.6, '1': 27.8, '1.1': 25.6, '1.2': 23.6}, 'Tl': {'0': 81, '0.1': 75.5, '0.2': 66.7, '0.3': 58.7, '0.4': 51.2, '0.5': 45.0, '0.6': 41.1, '0.7': 37.4, '0.8': 34.1, '0.9': 31.1, '1': 28.3, '1.1': 26.0, '1.2': 24.1}, 'Pb': {'0': 82, '0.1': 76.5, '0.2': 67.5, '0.3': 59.5, '0.4': 51.9, '0.5': 45.7, '0.6': 41.6, '0.7': 37.9, '0.8': 34.6, '0.9': 31.5, '1': 28.8, '1.1': 26.4, '1.2': 24.5}, 'Bi': {'0': 83, '0.1': 77.5, '0.2': 68.4, '0.3': 60.4, '0.4': 52.7, '0.5': 46.4, '0.6': 42.2, '0.7': 38.5, '0.8': 35.1, '0.9': 32.0, '1': 29.2, '1.1': 26.8, '1.2': 24.8}, 'Po': {'0': 84, '0.1': 78.4, '0.2': 69.4, '0.3': 61.3, '0.4': 53.5, '0.5': 47.1, '0.6': 42.8, '0.7': 39.1, '0.8': 35.6, '0.9': 32.6, '1': 29.7, '1.1': 27.2, '1.2': 25.2}, 'At': {'0': 85, '0.1': 79.4, '0.2': 70.3, '0.3': 62.1, '0.4': 54.2, '0.5': 47.7, '0.6': 43.4, '0.7': 39.6, '0.8': 36.2, '0.9': 33.1, '1': 30.1, '1.1': 27.6, '1.2': 25.6}, 'Rn': {'0': 86, '0.1': 80.3, '0.2': 71.3, '0.3': 63.0, '0.4': 55.1, '0.5': 48.4, '0.6': 44.0, '0.7': 40.2, '0.8': 36.8, '0.9': 33.5, '1': 30.5, '1.1': 28.0, '1.2': 26.0}, 'Fr': {'0': 87, '0.1': 81.3, '0.2': 72.2, '0.3': 63.8, '0.4': 55.8, '0.5': 49.1, '0.6': 44.5, '0.7': 40.7, '0.8': 37.3, '0.9': 34.0, '1': 31.0, '1.1': 28.4, '1.2': 26.4}, 'Ra': {'0': 88, '0.1': 82.2, '0.2': 73.2, '0.3': 64.6, '0.4': 56.5, '0.5': 49.8, '0.6': 45.1, '0.7': 41.3, '0.8': 37.8, '0.9': 34.6, '1': 31.5, '1.1': 28.8, '1.2': 26.7}, 'Ac': {'0': 89, '0.1': 83.2, '0.2': 74.1, '0.3': 65.5, '0.4': 57.3, '0.5': 50.4, '0.6': 45.8, '0.7': 41.8, '0.8': 38.3, '0.9': 35.1, '1': 32.0, '1.1': 29.2, '1.2': 27.1}, 'Th': {'0': 90, '0.1': 84.1, '0.2': 75.1, '0.3': 66.3, '0.4': 58.1, '0.5': 51.1, '0.6': 46.5, '0.7': 42.4, '0.8': 38.8, '0.9': 35.5, '1': 32.4, '1.1': 29.6, '1.2': 27.5}, 'Pa': {'0': 91, '0.1': 85.1, '0.2': 76.0, '0.3': 67.1, '0.4': 58.8, '0.5': 51.7, '0.6': 47.1, '0.7': 43.0, '0.8': 39.3, '0.9': 36.0, '1': 32.8, '1.1': 30.1, '1.2': 27.9}, 'U': {'0': 92, '0.1': 86.0, '0.2': 76.9, '0.3': 67.9, '0.4': 59.6, '0.5': 52.4, '0.6': 47.7, '0.7': 43.5, '0.8': 39.8, '0.9': 36.5, '1': 33.3, '1.1': 30.6, '1.2': 28.3}}
    return _dict


############ for peak convolution ############ 

def normal_cdf(x, y, mu):
    cdf = np.zeros_like(y)
    mask = x < mu
    cdf[mask] = np.cumsum(y[mask])
    return cdf

def axial_div(x, mu,L = 500, H = 100 / 2,S = 50 / 2 ):
    # Van Laar, B., & Yelon, W. B. (1984). The peak in neutron powder diffraction. Journal of Applied crystallography, 17(2), 47-54.
    axial_divergence = np.zeros_like(x)  # Initialize axial_divergence to zeros
    
    valid_indices = x <= mu  # Identify valid indices where x <= mu
    x_valid = x[valid_indices]  # Get valid x values
    
    h = L * np.sqrt((np.cos(np.radians(x_valid)) / np.cos(np.radians(mu)))**2 - 1)  # Calculate h

    W = np.where((H - S <= h) & (h <= H + S), H + S - h, 0)  # Calculate W for valid h

    axial_divergence[valid_indices] = L / (2 * H * S * h * np.cos(np.radians(x_valid))) * W  # Compute axial divergence for valid x values
    
    axial_divergence = axial_divergence/axial_divergence.max() # in case numerical err
    asym = normal_cdf(x, axial_divergence, mu)
    return asym

def slit_peak(x, mu,width=.1):
    height = 1/width 

    slit = np.where((x >= mu - width / 2) & (x <= mu + width / 2), height, 0)  # Create square wave function
    
    return slit

def gaussian(x, mu,sigma2=0.001):
    # sigma2 is lattice distortion
    sigma = np.sqrt(sigma2)
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)


def Voigt(x, mu, gamma, sigma2):

    z = ((x - mu) + 1j * gamma) / (np.sqrt(sigma2) * np.sqrt(2))  # Calculate complex parameter z
    Voigt_profile = np.real(wofz(z) / (np.sqrt(sigma2) * np.sqrt(2 * np.pi)))  # Calculate Voigt profile
    
    return Voigt_profile 


def map_int(peak, x, twotheta):
    y_twotheta = np.zeros_like(twotheta) # Initialize y_twotheta array
    _x = x[(x >= twotheta[0]) & (x <= twotheta[-1])]
    _peak = peak[(x >= twotheta[0]) & (x <= twotheta[-1])]
    for angle in range(len(_x)):
        index = np.argmin(np.abs( twotheta- _x[angle]))  # Find index for each angle
        if index.size > 0:  # Check if indices are not empty
            y_twotheta[index] = _peak[angle]  # Map peak intensity
    
    return y_twotheta

def combined_peak(twotheta, Weight, mu, gamma, sigma2, L,H,S,step=0.02):
    # Determine l_gap based on mu value
    if mu <= 10:
        l_gap = 7.8
    elif 10 < mu <= 15:
        l_gap = 10
    elif 15 < mu <= 20:
        l_gap = 15
    elif 20 < mu <= 30:
        l_gap = 20
    else:
        l_gap = 30
    
    # Ensure mu-l_gap and mu+l_gap are recorded in twotheta or its extension
    x = np.arange(np.round(mu - l_gap, 2), np.round(mu + l_gap, 2), step)
    
    # Calculate individual peaks
    voigt = Voigt(x, mu, gamma, sigma2)
    axial = axial_div(x, mu,L,H,S)
    slit = slit_peak(x, mu)
    distor = gaussian(x, mu)
    
    # Convolve the peaks
    combined = np.convolve(voigt, axial, mode='same')
    combined = np.convolve(combined, slit, mode='same')
    combined = np.convolve(combined, distor, mode='same')
    combined /= np.sum(combined) * step  # Normalize peak and apply weight
    
    # Map the peak to the original locations
    return map_int(combined, x, twotheta)*Weight