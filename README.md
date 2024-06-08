# SimXRD-4M
## database | [benchmark](https://github.com/compasszzn/XRDBench)

**Open Source:**  SimXRD-4M is freely available on our website (http://simxrd.caobin.asia/).

**Data Description:** Crystalline materials are categorized into 230 space groups, each representing a distinct symmetry class. XRD spectral data, which correspond to the crystal structure, serve as vital tools for studying these materials. However, spectral data are influenced by various factors such as the testing environment (instrumentation), light source (X-ray), and sample characteristics (grain size, orientation, etc.). Consequently, they exhibit varying characteristics, including changes in intensity values, peak broadening, etc., posing challenges for accurate phase identification. This database aims to facilitate model training by providing diffraction spectrum data under diverse environmental conditions. The ultimate goal is for the model to accurately identify the correct space group based on spectral data.

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

## acquire the review data by Croissant

+ [review data](https://huggingface.co/datasets/caobin/SimXRDreview)


``` javascript

import mlcroissant as mlc
url = "https://huggingface.co/datasets/caobin/SimXRDreview/raw/main/simxrd_croissant.json"


dataset_info = mlc.Dataset(url).metadata.to_json
print(dataset_info)
```

