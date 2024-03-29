# simXRDdatabase

## XRD2db

each crystal is simulated 50 times and saved sequentially, thus the first 50 spectra are same crysal and the next 50 spectra are other crysal and so on. 

``` javascript
from ase.db import connect

databs = connect("./binxrd.db") 

ids = [1,2,3] # id list
atom_list = [['C', 'H', 'O'],] # element list
latt_dis = [str([1, 2, 3, 4, 5, 6]),] # dis list, str
_int = [str([1, 2, 3, 4, 5, 6]),] # intensity list, str
_target = [str([221,1])] # target list, space group and crystal system
_chem_form = [str(PbSO4),] # chemical formula list
_simulation_param = [str(12552135)] # simulation parameters 

for id in ids:
    index = id - 1 
    atoms = Atoms(atom_list[index])
    databs.write(atoms=atoms, latt_dis=_dis[index], intensity=_int[index],tager=_target[index],
    chem_form=_chem_form[index], simulation_param=_simulation_param[index])

```

## db2XRD
``` javascript
databs = connect("./binxrd.db")

for row in databs.select():
    element = atoms.get_chemical_symbols()
    latt_dis = eval(getattr(row, 'latt_dis'))
    intensity = eval(getattr(row, 'intensity'))
    spg = eval(getattr(row, 'tager'))[0]
    crysystem = eval(getattr(row, 'tager'))[1]

    # element, a list, e.g., ['C', 'H', 'O']
    # latt_dis, a list, lattice plane distances
    # intensity, a list, diffraction intensity
    # spg, int, space group number
    # crysystem, int, crystal system 
```