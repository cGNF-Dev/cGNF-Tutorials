import numpy as np
np.set_printoptions(precision=3, suppress=None)  # Sets print options for numpy array outputs.
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'  # Setting the environment variable to choose the first GPU for CUDA to use.

base_path = 'C:\\Users\\Geoffrey Wodtke\\Dropbox\\D\\projects\\causal_normalizing_flows\\programs\\cGNF_tutorials'  # Define the base path for file operations.
folder = '_PSE'  # Define the folder where files will be stored.
path = os.path.join(base_path, folder, '')  # Combines the base path and folder into a complete path.
dataset_name = 'PSE_20k'  # Define the name of the dataset.

if not (os.path.isdir(path)):  # Checks if a directory with the name 'path' exists.
    os.makedirs(path)  # If not, creates a new directory with this name. This is where the logs and model weights will be saved.

## DATA SIMULATION
obs = 20000  # Sets the number of observations.

np.random.seed(2813308004) # Sets the seed for simulation.

C = np.random.binomial(n=1, p=0.5, size=obs)

epsilon_A = np.random.normal(0, 1, obs)
epsilon_L = np.random.logistic(0, 1, obs)
epsilon_M = np.random.laplace(0, 1, obs)
epsilon_Y = np.random.normal(0, 1, obs)

A = 0.2*C + epsilon_A
L = 0.1*A + 0.2*C + epsilon_L
M = 0.1*A + 0.1*C + 0.2*L + epsilon_M
Y = 0.1*A + 0.1*C + 0.2*M + 0.2*L + epsilon_Y*(1+0.2*C)

df = pd.DataFrame({'C': C, 'A': A, 'L': L, 'M': M, 'Y': Y})

df_filename = path + dataset_name + '.csv'
df.to_csv(df_filename, index=False)

## DAG SPECIFICATION
import collections.abc
collections.Iterable = collections.abc.Iterable
import networkx as nx
from causalgraphicalmodels import CausalGraphicalModel

simDAG = CausalGraphicalModel(
    nodes = ["C", "A", "L", "M", "Y"],
    edges = [("C", "A"), ("C", "L"), ("C", "M"), ("C", "Y"),
             ("A", "L"), ("A", "M"), ("A", "Y"),
             ("L", "M"), ("L", "Y"),
             ("M", "Y")])

print(simDAG.draw())

df_cDAG = nx.to_pandas_adjacency(simDAG.dag, dtype=int) # Converts the DAG to a pandas adjacency matrix.

print("------- Adjacency Matrix -------")
print(df_cDAG)

df_cDAG.to_csv(path + dataset_name + '_DAG.csv')

## FUNCTION HYPER-PARAMETER SPECIFICATION
process_args = {
    'test_size': 0.2, 
    'cat_var': ['C'], 
}

train_args = {
    'model_name': '20k',
    'trn_batch_size': 128, 
    'val_batch_size': 2048, 
    'learning_rate': 1e-4, 
    'seed': 8675309,
    'nb_epoch': 50000,
    'nb_estop': 50, 
    'val_freq': 1,
    'emb_net': [90, 80, 70, 60, 50],
    'int_net': [50, 40, 30, 20, 10]
}

sim_args1 = {
    'model_name': '20k',
    'treatment': 'A',
    'cat_list': [0,1],
    'mediator': ['L'],
    'outcome': 'M',
    'inv_datafile_name': 'med'
}

sim_args2 = {
    'model_name': '20k',
    'treatment': 'A',
    'cat_list': [0,1],
    'mediator': ['L', 'M'],
    'outcome': 'Y',
    'inv_datafile_name': 'pse'
}

## BOOTSTRAP INFERENCE
from cGNF import bootstrap
final_result = bootstrap(
    n_iterations=10,  # Number of bootstrap iterations
    num_cores_reserve=2,  # Number of cores to reserve
    base_path=path,
    folder_name='bootstrap_20k',
    dataset_name=dataset_name,
    dag_name=dataset_name + '_DAG',
    process_args=process_args,
    train_args=train_args,
    sim_args_list=[sim_args1, sim_args2]
)

## EFFECT ESTIMATION
final_result['ATE (A->Y)'] = final_result['E[Y(A=1)]'] - final_result['E[Y(A=0)]']
final_result['PSE (A->Y)'] = final_result['E[Y(A=1, L(A=0), M(A=0))]'] - final_result['E[Y(A=0)]']
final_result['PSE (A->L->Y)'] = final_result['E[Y(A=1)]'] - final_result['E[Y(A=1, L(A=0))]']
final_result['PSE (A->M->Y)'] = final_result['E[Y(A=1, L(A=0))]'] - final_result['E[Y(A=1, L(A=0), M(A=0))]']
final_result['ATE (A->M)'] = final_result['E[M(A=1)]'] - final_result['E[M(A=0)]']
final_result['NDE'] = final_result['E[M(A=1, L(A=0))]'] - final_result['E[M(A=0)]']
final_result['NIE'] = final_result['E[M(A=1)]'] - final_result['E[M(A=1, L(A=0))]']

final_results_df = final_result[[
    'ATE (A->Y)', 'PSE (A->Y)', 'PSE (A->L->Y)', 'PSE (A->M->Y)',
    'ATE (A->M)', 'NDE', 'NIE'
]]

## CONFIDENCE INTERVAL ESTIMATION
percentiles = {}
percentile_values = [10, 90]

outcome_columns = [col for col in final_results_df]

for column in outcome_columns:
    percentiles[column] = {}
    for p in percentile_values:
        percentiles[column][f'{p}th percentile'] = final_results_df[column].quantile(p/100)

# Convert the percentiles dictionary to a DataFrame
percentiles_df = pd.DataFrame(percentiles).transpose()

# Save the percentiles to a new CSV file
percentiles_csv_path = path + f'{dataset_name}_80%CI.csv'
percentiles_df.to_csv(percentiles_csv_path)
