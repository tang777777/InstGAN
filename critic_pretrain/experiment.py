import os


import torch
import numpy as np
import random
import pandas as pd
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from rdkit import Chem
from rdkit.Chem import PandasTools, QED, Descriptors, rdMolDescriptors
from training_process import FACGAN
import shutil
import torch.backends.cudnn
from mol_metrics import *

if __name__ == '__main__':

    seeds = 7

    # basic + tensorflow + torch 
    def seedEverything(seed=2):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'




    # load data

    
    train_smiles = []
    with open('../../zinc250k.csv', "r") as f:
        for line in f.readlines()[1:]:
            train_smiles.append(line.split(",")[1])
    # Training canonical SMILES
    data0 = [Chem.MolToSmiles(Chem.MolFromSmiles(sm)) for sm in train_smiles]
    data0 = list(set(data0))

    smiles_only = data0


    df_smiles = pd.Series(data0)

    # Convert SMILES to Mol
    df_mols = df_smiles.apply(Chem.MolFromSmiles)



    # qed (drug-likeness)
    Qed = df_mols.apply(QED.qed).rename('QED')

    # solubility
    Solubility = df_mols.apply(solubility).rename('Solubility')

    # synthesizability
    Sa = df_mols.apply(SA).rename('SA')

    # Combine the data frames into one
    combined_data = pd.concat([df_smiles.rename('SMILES'), Qed, Solubility, Sa], axis=1)
    
    # Transfrom dataframe into dictionary
    data = combined_data.to_dict('records')
    #print('data: ',data)
    
    def training(seed=2, flag='Pretrain', steps=60000):
        seedEverything(seed=seed)

        FACGAN_SMILES = FACGAN(smiles_only, latent_dim_size=256, lr_critic=2e-4, device='cuda:0', RL_Flag=flag)

        # load dataset
        loader = FACGAN_SMILES.dataloader_creation(data, batch_size=1024, shuffle=True, num_workers=0)
        import time
        start_time = time.time()

        # training
        FACGAN_SMILES.train_n(smiles_only, loader, max_step=steps, evaluate_every=100)

        end_time = time.time()
        runing_time = (end_time - start_time) / 60
        print('\n')
        print(f"runing time: {runing_time:.2f} minutes")

        import logging
        with open('logs/training_log.txt', 'a') as f:
            logging.basicConfig(filename='logs/training_log.txt', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

        logging.info('\n')
        logging.info(f"runing time: {runing_time:.2f} minutes")
        # Close the file handle
        logging.shutdown()

        if not os.path.exists(f"Results/{flag}/{steps}/logs"):
            os.makedirs(f"Results/{flag}/{steps}/logs")
        for file in os.listdir('logs'): 
            shutil.move(os.path.join('logs', file), f"Results/{flag}/{steps}/logs")

        if not os.path.exists(f"Results/{flag}/{steps}/models"):
            os.makedirs(f"Results/{flag}/{steps}/models")
        for file in os.listdir('models'): 
            shutil.move(os.path.join('models', file), f"Results/{flag}/{steps}/models")

        FACGAN_SMILES.eval()

        print('ok')


    
    s = 6
    training(seed=2, flag='QED', steps=s*10000)
    training(seed=2, flag='Solubility', steps=s*10000)
    training(seed=2, flag='SA', steps=s*10000)
