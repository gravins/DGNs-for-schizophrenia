# Drug repurposing for schizophrenia through Deep Graph Networks
This repository provides a reference implementation of our paper **Controlling astrocyte-mediated synaptic pruning signals for schizophrenia drug repurposing with Deep Graph Networks**.

## Requirements
_Note: we assume Miniconda/Anaconda is installed, otherwise see this [link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) for correct installation. The proper Python version is installed during the first step of the following procedure._

1. Install the required packages and create the environment
    - ``` conda env create -f env.yml ```

2. Activate the environment
    - ``` conda activate schizophrenia ```

## Experiments
- To rebuild the dataset run:

    ` python3 preprocessing.py `

- To perform the model selection with k-fold cross validation run: 

    ` python3 main_model_selection.py `
 
- To perform risk assessment of the model on the held-out test set run: 

    ` python3 main_risk_assessment.py `

- To perform the prediction of a new set of molecules, i.e., for biological validation purposes, run:

    1) ` python3 preprocessing.py --bioval --source_path <source_path_of_data>`
    2) ` python3 main_bioval_prediction.py --model_path <saved_model_path> --bioval_data_path <data_path>`
