# README for MSc Artificial Intelligence final project:
# (Dis)agreement detection in social media using Signed Graph Convolutional Network with NER and common noun stance integration by Bartosz Watrobinski and Haris Bin Zia
# The project code contains the following elements:

# train.py: the code containing necessary function calls to run the model and report the results. Includes novel implementations by project authors as well as original implementation by #Lorge et al. (2024).
# helper_functions.py: the file containing functions necessary for preprocessign the GNN and dataset for running it. It includes the novel functionalities by the project authors and the # #original code implementation by Lorge et al. (2024).
# dataset.py: the file containing code for preprocessing textual data from the dataset and constructing signed bipartite graphs.
# models.py: the file including the models code for Signed Convolutional Weighted Graph and STEntConv model developed by Lorge et al. (2024)
#train.csv: contains dataset part for training the model.
#dev.csv: contains dataset part for model evaluation
#dev.csv contains dataset part for model testing.
# pretrained_bert: directory containing fine-tuned BERT on the whole DEBAGREEMENT dataset
# json files which contain precomputed edges


# to run the code please put all directory content in the same directory, open BERT_Training.ipynb, run the resulting code there and run train.py
# In case of any issues please contact Bartosz Watrobinski

# Requirements: 
#python 3.10.13
#cuda version 11.8
#gensim==4.2.0
#pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
#pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
#torch-scatter
#torch-sparse
#torch-geometric==2.4.0
#torch
#transformers
#sentence-transformers
#spacy
#nltk
#python -m spacy download en_core_web_md
#python -m spacy download en_core_web_sm
#python -m nltk.downloader wordnet
#python -m nltk.downloader omw-1.4
#tqdm
#pandas
#scikit-learn
#scipy
#kneed
#
#
#
#
#
#
#
#
#
#
#
#
#
#
