# SimXRD-4M
## database | [benchmark](https://github.com/compasszzn/XRDBench)

**Open Source:**  SimXRD-4M is freely available on our website (http://simxrd.caobin.asia/).

**Data Description:** Crystalline materials are categorized into 230 space groups, each representing a distinct symmetry class. XRD spectral data, which correspond to the crystal structure, serve as vital tools for studying these materials. However, spectral data are influenced by various factors such as the testing environment (instrumentation), light source (X-ray), and sample characteristics (grain size, orientation, etc.). Consequently, they exhibit varying characteristics, including changes in intensity values, peak broadening, etc., posing challenges for accurate phase identification. This database aims to facilitate model training by providing diffraction spectrum data under diverse environmental conditions. The ultimate goal is for the model to accurately identify the correct space group based on spectral data.

## paper 
+ [arxiv](https://arxiv.org/pdf/2406.15469v1)
  
``` javascript
@misc{cao2024simxrd4mbigsimulatedxray,
      title={SimXRD-4M: Big Simulated X-ray Diffraction Data Accelerate the Crystalline Symmetry Classification}, 
      author={Bin Cao and Yang Liu and Zinan Zheng and Ruifeng Tan and Jia Li and Tong-yi Zhang},
      year={2024},
      eprint={2406.15469},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2406.15469}, 
}

```
## installation

You'll need to install the following libraries for processing db file:

- ase
- tqdm
  

## kaggle competition announcement

To benchmark advanced models and further their development, we are launching a Kaggle competition for space group classification. Participants are invited to upload their predictions based on the [testNOtgt data](https://github.com/Bin-Cao/SimXRD/tree/main/testNOtgt_db) using their trained models. Submit your results on the [Leaderboard](https://www.kaggle.com/competitions/simxrd/leaderboard). 

<img width="973" alt="Screenshot 2024-06-13 at 22 15 21" src="https://github.com/Bin-Cao/SimXRD/assets/86995074/e125623f-d695-4624-b6fc-3d0604dc2846">

For more detailed information, please visit the [Kaggle competition page](https://www.kaggle.com/competitions/simxrd).

## readin data
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
# 1. Point to the Croissant file
    import mlcroissant as mlc
    url = "https://huggingface.co/datasets/caobin/SimXRDreview/raw/main/simxrd_croissant.json"

# 2. Inspect metadata
  dataset_info = mlc.Dataset(url).metadata.to_json
  print(dataset_info)

  from dataset.parse import load_dataset,bar_progress # defined in our github : https://github.com/compasszzn/XRDBench/blob/main/dataset/parse.py
  for file_info in dataset_info['distribution']:
      wget.download(file_info['contentUrl'], './', bar=bar_progress)

# 3. Use Croissant dataset in your ML workload
  train_loader = DataLoader(load_dataset(name='train.tfrecord'), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
  val_loader = DataLoader(load_dataset(name='val.tfrecord'), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,drop_last=False)
  test_loader = DataLoader(load_dataset(name='test.tfrecord'), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,drop_last=False)
```

