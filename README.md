# ALL DATA are OPEN-sourced on ONE drive. 
## dataset
**Data Description:** Crystalline materials are categorized into 230 space groups, each representing a distinct symmetry class. XRD spectral data, which correspond to the crystal structure, serve as vital tools for studying these materials. However, spectral data are influenced by various factors such as the testing environment (instrumentation), light source (X-ray), and sample characteristics (grain size, orientation, etc.). Consequently, they exhibit varying characteristics, including changes in intensity values, peak broadening, etc., posing challenges for accurate phase identification. This database aims to facilitate model training by providing diffraction spectrum data under diverse environmental conditions. The ultimate goal is for the model to accurately identify the correct space group based on spectral data.

**Task Description:** Input consists of sequence data for 230 classification problems. For additional data requirements, please reach out to the author.


## training data
+ [train_1_binxrd.db](https://github.com/Bin-Cao/SimXRDdb/tree/main/train_db) **119,569*5** data
+ [train_2_binxrd.db](https://github.com/Bin-Cao/SimXRDdb/tree/main/train_db) **119,569*5** data
+ [train_3_binxrd.db](https://github.com/Bin-Cao/SimXRDdb/tree/main/train_db) **119,569*5** data
+ [train_4_binxrd.db](https://github.com/Bin-Cao/SimXRDdb/tree/main/train_db) **119,569*5** data
+ [train_5_binxrd.db](https://github.com/Bin-Cao/SimXRDdb/tree/main/train_db) **119,569*5** data
+ [train_6_binxrd.db](https://github.com/Bin-Cao/SimXRDdb/tree/main/train_db) **119,569*5** data

(**119,569*30** data of XRD spectra, ***X** denotes the simulation of one crystal under varied conditions, resulting in X distinct spectra.)
  
## testing data
+ [test_binxrd.db](https://github.com/Bin-Cao/SimXRDdb/tree/main/test_db) **119,569*1** data

We recommend utilizing this **testing dataset (119,569 entries)** to compare the predictive capabilities of different models.

**Please email me your name, organization, and the purpose of your application in order to receive the password.**

(***X** denotes the simulation of one crystal under varied conditions, resulting in X distinct spectra.)
## data source
The dataset, containing 119,569*30 data of XRD spectra and chemical composition, is retrieved from the [Materials Project (MP) database](https://materialsproject.org) and simulated by [WPEM](https://github.com/WPEM)

## cite
Cao, B. (2024). **SimXRDdb: X-ray diffraction simulation spectra database.** Retrieved from https://github.com/Bin-Cao/SimXRDdb.

## contributing 
Contribution and suggestions are always welcome. In addition, we are also looking for research collaborations. You can submit issues for suggestions, questions, bugs, and feature requests, or submit pull requests to contribute directly. You can also contact the authors for research collaboration.

## xrd2db

**Each crystal is simulated 30 times and saved sequentially. Therefore, the first 30 spectra belong to the same crystal, followed by the next 30 spectra representing another crystal, and so forth.**


``` javascript
from ase.db import connect
from ase import Atoms

databs = connect("./binxrd.db") 

atom_list = [['C', 'H', 'O'],] # element list
latt_dis = [str([1, 2, 3, 4, 5, 6]),] # dis list, str
_int = [str([1, 2, 3, 4, 5, 6]),] # intensity list, str
_target = [str([221,1])] # target list, space group and crystal system
_chem_form = [str(PbSO4),] # chemical formula list
_simulation_param = [str(10,20%,0.3,1.5)] # simulation parameters : GrainSize,orientation,thermo_vib,zero_shift

for id in ids:
    index = id - 1 
    atoms = Atoms(atom_list[index])
    databs.write(atoms=atoms, latt_dis=_dis[index], intensity=_int[index],tager=_target[index],
    chem_form=_chem_form[index], simulation_param=_simulation_param[index])

```

## db2xrd
``` javascript
from ase.db import connect
databs = connect("./binxrd.db")

for row in databs.select():
    atoms = row.toatoms()
    element = atoms.get_chemical_symbols()
    latt_dis = eval(getattr(row, 'latt_dis'))
    intensity = eval(getattr(row, 'intensity'))
    spg = eval(getattr(row, 'tager'))[0]
    crysystem = eval(getattr(row, 'tager'))[1]

    # element, a list, e.g., ['C', 'H', 'O']
    # latt_dis, a list, lattice plane distances
    # intensity, a list, diffraction intensity
    # spg, int, space group number
    # crysystem, int, crystal system number
```


