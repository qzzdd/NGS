# NGS
The author implementation of "[Explainable Graph-based Fraud Detection via Neural Meta-graph Search](https://dl.acm.org/doi/abs/10.1145/3511808.3557598)" (CIKM 2022).

## Requirements

```
numpy==1.23.5
scikit_learn==0.24.2
scipy==1.9.1
torch==1.9.0
```

## Dataset
YelpChi and Amazon can be downloaded from [here](https://github.com/safe-graph/RioGNN/tree/main/data).

Please unzip the downloaded files, and put the `"Amazon.mat"` and `"YelpChi.mat"` to the `"/data"` directory .

## Usage

```sh
python run_all.py --steps 4 4 4 4 --dataset amazon
python run_all.py --steps 4 4 4 4 --dataset yelp
```

## Citation

```
@inproceedings{qin2022explainable,
    author = {Qin, Zidi and Liu, Yang and He, Qing and Ao, Xiang},
    title = {Explainable Graph-Based Fraud Detection via Neural Meta-Graph Search},
    year = {2022},
    booktitle = {Proceedings of the 31st ACM International Conference on Information and Knowledge Management},
    pages = {4414–4418},
}
```


