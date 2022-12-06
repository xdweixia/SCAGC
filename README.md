# SCAGC
This repo provides a demo for the TMM-2022 article: [Self-consistent Contrastive Attributed Graph Clustering with Pseudo-label Prompt](https://ieeexplore.ieee.org/document/9914670) on the ACM dataset.

## Requirements
* `Python 3.6`
* `PyTorch 1.6.0`
* `PyTorch Geometric 1.6.1`

## Training
###-Step 1: Warm-Up:
```
python \ACM-Pretrain-1\Pre_train_1.py
```
Once the training is finished, then remember copy the generated `Pre_train_1.pkl` to the folder `\ACM-Pretrain-2`:
```
python \ACM-Pretrain-2\Pretrain_2.py
```
Once the training is finished, then remember copy the generated `ACM_pretrain.pkl` to the folder `\ACM-Final`:

###-Step 2: Clustering:
```
python \ACM-Final\run_cluster.py
```

## Acknowledgements
Some codes are adapted from [GCA](https://github.com/CRIPAC-DIG/GCA) and [SupContrast](https://github.com/HobbitLong/SupContrast). We thank them for their excellent projects.

## Contact
If you have any problem about our code, feel free to contact xd.weixia@gmail.com or describe your problem in Issues.
