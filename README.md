# [(Dis)agreement detection in social media using Signed Graph Convolutional Network with NER and common noun stance integration](https://drive.google.com/file/d/1_CmrsM4pAPiXD978Eg8mxjxM7kyb32tz/view?usp=sharing)

## Setup
* Move to your desired local folder
* Open terminal: `git clone https://github.com/bartoszwatrobinski/postgraduate-research-project`
* Create a new environment `python3 -m venv venv`
* Activate the environment, on Windows `venv\Scripts\activate`, on Mac `source venv/bin/activate`.
* Install required dependencies `pip install -r requirements.txt`
* Install additional resources:
  ```bash
  python -m spacy download en_core_web_md
  python -m spacy download en_core_web_sm
  python -m nltk.downloader wordnet
  python -m nltk.downloader omw-1.4
* Open the `BERT_Training.ipynb` file in Jupyter Notebook or your preferred IDE that supports .ipynb files.
* Execute the `train.py` script


## Overview

<p align="center">
    <img width=500px src='/images/1.png'>
</p>

<p align="center">
    <img width=500px src='/images/2.png'>
</p>

<p align="center">
    <img width=500px src='/images/3.png'>
</p>

<p align="center">
    <img width=500px src='/images/4.png'>
</p>

<p align="center">
    <img width=500px src='/images/5.png'>
</p>

<p align="center">
    <img width=500px src='/images/6.png'>
</p>

<p align="center">
    <img width=500px src='/images/7.png'>
</p>

<p align="center">
    <img width=500px src='/images/8.png'>
</p>

<p align="center">
    <img width=500px src='/images/9.png'>
</p>

<p align="center">
    <img width=500px src='/images/10.png'>
</p>

<p align="center">
    <img width=500px src='/images/11.png'>
</p>

## The project includes the following files:

* train.py: the code containing necessary function calls to run the model and report the results. Includes novel implementations by project authors as well as original implementation by #Lorge et al. (2024).
* helper_functions.py: the file containing functions necessary for preprocessign the GNN and dataset for running it. It includes the novel functionalities by the project authors and the original code implementation by Lorge et al. (2024).
* dataset.py: the file containing code for preprocessing textual data from the dataset and constructing signed bipartite graphs.
* models.py: the file including the models code for Signed Convolutional Weighted Graph and STEntConv model developed by Lorge et al. (2024)
* train.csv: contains dataset part for training the model.
* dev.csv: contains dataset part for model evaluation
* dev.csv contains dataset part for model testing.
* pretrained_bert: directory containing fine-tuned BERT on the whole DEBAGREEMENT dataset
* json files which contain precomputed edges



## Note
* This is a postgraduate research project by Bartosz Watrobinski under the supervision of Haris Bin Zia at Queen Mary University of London.
* The associated project description and evaluation can be found [here](https://drive.google.com/file/d/1_CmrsM4pAPiXD978Eg8mxjxM7kyb32tz/view?usp=sharing)
* In case of any issues please contact Bartosz Watrobinski

