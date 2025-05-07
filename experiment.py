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
import logging


if __name__ == '__main__':

    seeds = 1

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

    
    # Path to the combined properties CSV file
    csv_file_path = 'chembl_selected_combined_properties.csv'

    # Read the combined data from the CSV file
    combined_data_df = pd.read_csv(csv_file_path)

    # If you want it as a list instead
    smiles_only = combined_data_df['SMILES'].tolist()


    # Combine the data frames into one
    combined_data = combined_data_df
    
    # Transfrom dataframe into dictionary
    data = combined_data.to_dict('records')
    #print('data: ',data)
    
    def training(seed=2, flag='Pretrain', Alpha=0.5, qed_w=0.5, drd2_w=0.5, ew=1e-3, gw=7.1e-5, lr=1e-4):
        seedEverything(seed=seed)

        FACGAN_SMILES = FACGAN(smiles_only, latent_dim_size=256, lr_generator=lr, lr_discriminator=lr, lr_critic=lr, device='cuda:0', EW=ew, SRW_D=gw, SRW_C=gw, RL_Flag=flag, Alpha_Initial=Alpha, CGW=0.5, QED_W=qed_w, Logp_W=0.0, SA_W=0.0, DRD2_W=drd2_w)

        # load dataset
        loader = FACGAN_SMILES.dataloader_creation(data, batch_size=256, shuffle=True, num_workers=0)
        import time
        start_time = time.time()

        # training
        FACGAN_SMILES.train_n(smiles_only, loader, max_step=5000, evaluate_every=100)

        end_time = time.time()
        runing_time = (end_time - start_time) / 60
        print('\n')
        print(f"runing time: {runing_time:.2f} minutes")

        
        # Set up logging
        logging.basicConfig(filename='logs/training_log.txt', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', filemode='a')
        logging.info('\n')
        logging.info(f"runing time: {runing_time:.2f} minutes")

        logging.info('\n')

        logging.info(f'Amount of data used for training: {len(data)}')

        # Close the file handle
        logging.shutdown()

        '''
        # move log file and best_generated_data to Results/{flag}/logs
        if not os.path.exists(f"Results/{flag}/logs"):
            os.makedirs(f"Results/{flag}/logs")
        for file in os.listdir('logs'): 
            shutil.move(os.path.join('logs', file), f"Results/{flag}/logs")

        # move all models to Results/{flag}/models
        if not os.path.exists(f"Results/{flag}/models"):
            os.makedirs(f"Results/{flag}/models")
        for file in os.listdir('models'): 
            shutil.move(os.path.join('models', file), f"Results/{flag}/models")
        '''



        if not os.path.exists(f"Results/{flag}/qed_w={qed_w}_drd2_w={drd2_w}/Alphas/{Alpha}/logs"):
            os.makedirs(f"Results/{flag}/qed_w={qed_w}_drd2_w={drd2_w}/Alphas/{Alpha}/logs")
        for file in os.listdir('logs'): 
            shutil.move(os.path.join('logs', file), f"Results/{flag}/qed_w={qed_w}_drd2_w={drd2_w}/Alphas/{Alpha}/logs")

        if not os.path.exists(f"Results/{flag}/qed_w={qed_w}_drd2_w={drd2_w}/Alphas/{Alpha}/models"):
            os.makedirs(f"Results/{flag}/qed_w={qed_w}_drd2_w={drd2_w}/Alphas/{Alpha}/models")
        for file in os.listdir('models'): 
            shutil.move(os.path.join('models', file), f"Results/{flag}/qed_w={qed_w}_drd2_w={drd2_w}/Alphas/{Alpha}/models")



        FACGAN_SMILES.eval()

    #training(seed=2, flag='QED_DRD2', Alpha=0.5, qed_w=0.5, drd2_w=0.5, ew=2.2e-2, gw=1e-5, lr=7e-6)
    #training(seed=2, flag='QED_DRD2', Alpha=0.5, qed_w=0.4, drd2_w=0.6, ew=2.2e-2, gw=1e-5, lr=7e-6)
    #training(seed=2, flag='QED_DRD2', Alpha=0.5, qed_w=0.3, drd2_w=0.7, ew=2.2e-2, gw=1e-5, lr=7e-6)
    #training(seed=2, flag='QED_DRD2', Alpha=0.5, qed_w=0.2, drd2_w=0.8, ew=2.2e-2, gw=1e-5, lr=7e-6)
    training(seed=2, flag='QED_DRD2', Alpha=0.5, qed_w=0.1, drd2_w=0.9, ew=2.2e-2, gw=1e-5, lr=7e-6)
    
    '''
    for i in range(11):
        alpha = i/10
        training(seed=2, flag='QED_DRD2', Alpha=alpha, ew=1e-3, gw=5e-5, lr=2e-4)
    '''