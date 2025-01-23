# SimXRD-4M ICLR 2025

## The Official Implementation of SimXRD | [Paper](https://openreview.net/forum?id=mkuB677eMM) | [DataBase](https://huggingface.co/AI4Spectro)|[Benchmark](https://github.com/compasszzn/XRDBench)

**Open Source:** SimXRD-4M is freely available on our website ([http://simxrd.caobin.asia/](http://simxrd.caobin.asia/)) & [Huggingface](https://huggingface.co/AI4Spectro).

**Data Description:** Crystals are categorized into 230 space groups, each representing a distinct symmetry catrgory. XRD patterns, which correspond to the crystal structure, serve as vital tools for studying these materials. However, XRD patterns are influenced by various factors such as the testing environment (instrumentation), light source (X-ray), and sample characteristics (grain size, orientation, etc.). Consequently, they exhibit varying characteristics, including changes in intensity values, peak broadening, etc., posing challenges for accurate phase identification. This database aims to facilitate model training by providing diffraction spectrum data under diverse environmental conditions. The ultimate goal is for the model to accurately identify the correct space group based on XRD patterns.

---

## Installation

You'll need to install the following libraries for processing the database file:

- ase
- tqdm

```bash
pip install ase tqdm
```

Kaggle Competition Announcement

To benchmark advanced models and further their development, we are launching a Kaggle competition for space group classification. Participants are invited to upload their predictions based on the testNOtgt data using their trained models. Submit your results on the Leaderboard.
For more detailed information, please visit the Kaggle competition page.

Reading Data
```Python
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

## Tutorials
- **training**: [model_tutorial](./tutorial/template.ipynb)
- **simulation**: [sim_tutorial](./sim/XRD.ipynb)
- **high-throughput simulation**: [HTsim_tutorial](./sim/tutorial_sim.ipynb)

(If you need the organized crystal database, please feel free to contact me to acquire it through a collaborative arrangement)

Dataset Distribution

Database: [test_binxrd]
Description: The test_binxrd database houses 119,569*2 X-ray diffraction (XRD) simulation spectra in d-I format. This dataset serves as a testing dataset, wherein each crystal corresponds to only one spectrum.

Database: [train_binxrd]
Description: The train_binxrd database houses 119,569*30 X-ray diffraction (XRD) simulation spectra in d-I format. This dataset serves as a training dataset, wherein each crystal corresponds to 5 spectra in each file.

Database: [val_binxrd]
Description: The val_binxrd database houses 119,569 X-ray diffraction (XRD) simulation spectra in d-I format. This dataset serves as a validation dataset, wherein each crystal corresponds to only one spectrum.

Database: [testNOtgt]
Description: The testNOtgt database houses 119,569 X-ray diffraction (XRD) simulation spectra in d-I format. This dataset serves as a testing dataset, which lacks a target variable and is randomly ordered.


Acquire Review Data by Croissant
Review Data
```Python
# 1. Point to the Croissant file
import mlcroissant as mlc
url = "https://huggingface.co/datasets/caobin/SimXRDreview/raw/main/simxrd_croissant.json"


# 2. Inspect metadata
dataset_info = mlc.Dataset(url).metadata.to_json()
print(dataset_info)

from dataset.parse import load_dataset, bar_progress  # defined in our github: https://github.com/compasszzn/XRDBench/blob/main/dataset/parse.py
for file_info in dataset_info['distribution']:
    wget.download(file_info['contentUrl'], './', bar=bar_progress)

# 3. Use Croissant dataset in your ML workload
from torch.utils.data import DataLoader

train_loader = DataLoader(load_dataset(name='train.tfrecord'), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(load_dataset(name='val.tfrecord'), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
test_loader = DataLoader(load_dataset(name='test.tfrecord'), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
```
