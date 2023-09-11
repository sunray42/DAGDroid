# DAGDroid
Source code for paper below:   
**DAGDroid: Android Malware Detection Method for Concept Drift Using Domain
Adversarial Graph Neural Networks**[paper]


## Dependencies
* Python 3.8.13
* PyTorch 1.10.1
* dgl 0.9.1
* tllib 0.4
* scikit-learn 1.0.2
* numpy 1.21.2

## Environment
1. Create the environment from the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```

2. Activate the new environment:
    ```bash
    conda activate dgl
    ```

## Dataset
Dataset comes from [TESSERACT: eliminating experimental bias in malware classification across space and time](https://dl.acm.org/doi/abs/10.5555/3361338.3361389).

## Project structure
```bash
<DAGDroid>
|-- checkpoints     # save model state dict and performance
|   |-- history_records
|   |__ reports.csv
|
|-- dataset.py      # construct customized DGL dataset class
|-- evoluNetwork.py     # construct Android evolutionary network
|-- model.py
|-- pretrain.py
|-- reweight.py
|-- utils.py
|-- train.py
|-- README.md
|__ environment.yml
```

## How to run
1. Construct Android evolutionary graph:
```python
python3 evoluNetwork.py
```

2. Pretrain a feature exactor and a classifier:
```python
nohup python3 pretrain.py >> nohup.out &
```

3. Train DAGDroid:
```python
nohup python3 train.py --load-model pretrain --lr 0.2 --trade-off 5 --inductive --batch-size 64 --iters-per-epoch 1000 --num-epochs 25 >> nohup.out &
```