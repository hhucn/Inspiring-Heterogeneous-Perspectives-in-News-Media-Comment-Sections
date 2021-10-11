# Inspiring Heterogeneous Perspectives in News Media Comment Sections


This repository contains the code for the experiment and evaluation of the paper "Inspiring Heterogeneous Perspectives in News Media Comment Sections". Additionally, it contains the jupyter notebooks used to compute the embeddings of the keywords of the articles and the comments in the dataset from the New York Times. 


## Experiment
The folder ``Experiment`` contains the code for the model of this paper, as well as the code to run the experiment.

Please note, that we do not include the embeddings of the dataset due to the size of the embedding files. 

If you want to reproduce our results, you can compute the embeddings yourself by running the jupyter notebooks in the folder ``GoogleColabNotebooks``.

To reproduce the experiment you have to run the ``main`` method in ``Experiment.py``.

## GoogleColabNotebooks

This folder contains the jupyter notebooks to compute the embeddings for the various models we have tested in our experiment. We computed the embeddings for the keywords of the articles and the embeddings for the comments for every model in seperated notebooks.

## Evaluation

The ``Evaluation`` folder contains the annotations for the evaluation of the model, the experiment results that were annotated, and a notebook with the evaluation of the annotation results.
