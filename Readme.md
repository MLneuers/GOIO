# GOIO: Generative Oversampling Approach to Class Imbalance and Overlap of Tabular Data

This repository contains the official implementation of "**GOIO: Generative Oversampling Approach to Class Imbalance and Overlap of Tabular Data**". 
# Setup Instructions

## Install the required packages

Python 3.10 or higher is required. Install the necessary Python packages by running the following command:

```
pip install -r requirements.txt  
```



# Running the experiments

Below are the steps to reproduce the experimental results.

## Data Preparation 

1. Prepare your dataset in the `./data/datasets/[NAME_OF_DATASET]` directory. The file should be in `.csv` or `.xlsx` format, named as `[NAME_OF_DATASET].csv` or `[NAME_OF_DATASET].xlsx`.
2. Use the following command to preprocess the data for GOIO training:

```
# Data preprocessing  
python main.py --dataname abalone_15 --method data --mode split  
```

3. To generate artificial data, use the following command:

Besides, the artificial data can be accessed through the following commands：

```
# Generate artificial data  
python main.py --method data --mode syn --means 0.5 --CR 5  
```

- The **level of class overlap** and **class imbalance** in the artificial data can be adjusted using the `--means` and `--CR` parameters, respectively.



## Training Models
To train the models in GOIO, follow these steps:

1. Train the MLVAE model:

```
# Train MLVAE  
python main.py --dataname abalone_15 --method MLVAE --mode train  
```

2. After the MLVAE model is trained, train the CLDM model:

```
# Train CLDM  
python main.py --dataname abalone_15 --method CLDM --mode train  
```

## Sample and Evaluation

To perform data synthesis and evaluate the methods, run the following commands:

1. Oversample the minority data:

```
# Oversampling  
python main.py --dataname abalone_15 --method CLDM --mode sample
```

2. Run the evaluation pipeline:

```
# Evaluation  
python main.py --dataname abalone_15 --method CLDM --mode eval  
```



## Acknowledgements

This project was developed based on the open-source work [TabSyn](https://github.com/amazon-science/tabsyn), which is licensed under the Apache License 2.0. We sincerely thank the authors for providing their code and inspiration, which served as a foundation for our work.

If you use this repository or parts of it, please consider citing the original TabSyn work as follows:

```
@inproceedings{tabsyn,
  title={Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space},
  author={Zhang, Hengrui and Zhang, Jiani and Srinivasan, Balasubramaniam and Shen, Zhengyuan and Qin, Xiao and Faloutsos, Christos and Rangwala, Huzefa and Karypis, George},
  booktitle={The twelfth International Conference on Learning Representations},
  year={2024}
}
```
