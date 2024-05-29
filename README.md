
## SimXRD-4M
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

## validation data
+ [val_binxrd.db](https://github.com/Bin-Cao/SimXRDdb/tree/main/val_db) **119,569*1** data

We recommend utilizing this **validation dataset (119,569 entries)** as the validation set.


## testing data
+ [test_binxrd.db](https://github.com/Bin-Cao/SimXRDdb/tree/main/test_db) **119,569*2** data

We recommend utilizing this **testing dataset (119,569*2 entries)** to compare the predictive capabilities of different models.


## testing data without target
+ [testNOtgt.db](https://github.com/Bin-Cao/SimXRDdb/tree/main/testNOtgt_db) **119,569*1** data

We highly recommend utilizing the **testNOtgt.db dataset**, comprising 119,569 entries, to assess the predictive generalizability of your model. This dataset lacks a target variable and is randomly ordered. Please feel free to email me your predictions (in npy or csv format), and I will evaluate the accuracy of your model accordingly.

(***X** denotes the simulation of one crystal under varied conditions, resulting in X distinct spectra.)

## data source
The dataset, containing 119,569*30 data of XRD spectra and chemical composition, is retrieved from the [Materials Project (MP) database](https://materialsproject.org) and simulated by [WPEM](https://github.com/WPEM)


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


