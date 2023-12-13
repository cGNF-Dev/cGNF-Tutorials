import numpy as np
np.set_printoptions(precision=3, suppress=None)  # Sets print options for numpy array outputs.
import pandas as pd 
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # Setting the environment variable to choose the first GPU for CUDA to use.

base_path = 'C:\\Users\\Geoffrey Wodtke\\Dropbox\\D\\projects\\causal_normalizing_flows\\programs\\cGNF_tutorials'  # Define the base path for file operations.
folder = '_CATE'  # Define the folder where files will be stored.
path = os.path.join(base_path, folder, '')  # Combines the base path and folder into a complete path.
dataset_name = 'CATE_20k'  # Define the name of the dataset.

if not (os.path.isdir(path)):  # Checks if a directory with the name 'path' exists.
    os.makedirs(path)  # If not, creates a new directory with this name. This is where the logs and model weights will be saved.

## DATA SIMULATION
obs = 20000  # Sets the number of observations.

np.random.seed(2813308004) # Sets the seed for simulation.

C = np.random.binomial(n=1, p=0.5, size=obs)

epsilon_A = np.random.logistic(0, 1, obs)
epsilon_Y = np.random.normal(0, 1, obs)

A = 0.3*C + epsilon_A
Y = 0.2*A + 0.4*C + epsilon_Y

df = pd.DataFrame({'C': C, 'A': A, 'Y': Y})

df_filename = path + dataset_name + '.csv'
df.to_csv(df_filename, index=False)

## DAG SPECIFICATION
import collections.abc
collections.Iterable = collections.abc.Iterable
import networkx as nx
from causalgraphicalmodels import CausalGraphicalModel

simDAG = CausalGraphicalModel(
    nodes = ["C", "A", "Y"],
    edges = [("C", "A"), ("C", "Y"),
             ("A", "Y")])

print(simDAG.draw())

df_cDAG = nx.to_pandas_adjacency(simDAG.dag, dtype=int) # Converts the DAG to a pandas adjacency matrix.

print("------- Adjacency Matrix -------")
print(df_cDAG) 

df_cDAG.to_csv(path + dataset_name + '_DAG.csv')

## DATA PREPROCESSING
from cGNF import process
process(path=path, dataset_name=dataset_name, dag_name=dataset_name + '_DAG', test_size=0.2, cat_var=['C'], seed=None)

## MODEL TRAINING
from cGNF import train
train(path=path, dataset_name=dataset_name, model_name='20k',
        trn_batch_size=128, val_batch_size=2048, learning_rate=1e-4, seed=8675309,
        nb_epoch=50000, nb_estop=50, val_freq=1,
        emb_net=[90, 80, 70, 60, 50],
        int_net=[50, 40, 30, 20, 10])

## POTENTIAL OUTCOME ESTIMATION
from cGNF import sim
sim(path=path, dataset_name=dataset_name, model_name='20k', n_mce_samples=50000, inv_datafile_name='sim_20k',
        treatment='A', cat_list=[0, 1], moderator='C', mediator=None, outcome='Y')
