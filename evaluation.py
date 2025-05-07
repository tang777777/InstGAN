import warnings
import numpy as np 
import pandas as pd
from rdkit import Chem
from rdkit import rdBase
from rdkit import DataStructs
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")
from rdkit.Chem import PandasTools
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
import os
from mol_metrics import *
import logging
from rdkit.Chem import PandasTools, QED, Descriptors, rdMolDescriptors




def mean_diversity(smiles):
    """
    args:
        - smiles: a list of SMILES strings

    returns:
        - average diversity score
    """
    scores = []
    df = pd.DataFrame({'smiles': smiles})
    PandasTools.AddMoleculeColumnToFrame(df, 'smiles', 'mol')
    fps = [GetMorganFingerprintAsBitVect(m, 4, nBits=2048) for m in df['mol'] if m is not None]
    for i in range(1, len(fps)):
        scores.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i], returnDistance=True))
    
    return np.mean(scores)

def evaluation(generated_smiles, training_smiles, log_file='logs/training_log.txt'):
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up logging
    logging.basicConfig(filename='logs/training_log.txt', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', filemode='a')

    """
    args:
        - generated_smiles: a list of generated SMILES strings
        - training_smiles: training SMILES dataset

    returns:
        - validity: ratio of valid SMILES strings in generated SMILES stirngs
        - uniqueness: ratio of unique SMILES strings in the valid SMILES strings
        - novelty: ratio of novel SMILES strings in the unique SMILES strings
        - diversity: diversity score of the novel SMILES strings
    """

    # SMILES to molecules
    generated_mols = np.array([Chem.MolFromSmiles(s) for s in generated_smiles if len(s.strip())])
    
    if len(generated_mols) == 0:
        print('No SMILES data is generated, please pre-train the generator again!')
        return

    else:

        train_smiles = training_smiles
        valid_smiles = [Chem.MolToSmiles(mol) for mol in generated_mols if mol != None and mol.GetNumAtoms() > 1 and Chem.MolToSmiles(mol) != ' ']
        unique_smiles = list(set(valid_smiles))
        novel_smiles = [smi for smi in unique_smiles if smi not in train_smiles]

        validity = len(valid_smiles)/len(generated_mols)
        if len(valid_smiles) == 0:
            valid_smiles.append('c1ccccc1')
        if len(unique_smiles) == 0:
            unique_smiles.append('c1ccccc1')
        uniqueness = len(unique_smiles)/len(valid_smiles)
        novelty = len(novel_smiles)/len(unique_smiles)
        diversity = mean_diversity(novel_smiles)
        total = validity*uniqueness*novelty
        
        
        # Convert SMILES to Mol objects and filter out invalid entries
        novel_data = pd.Series(novel_smiles).apply(Chem.MolFromSmiles).dropna()
        novel_data = novel_data[novel_data.apply(lambda x: x.GetNumAtoms() > 1)]
        

        if not novel_data.empty:
            # Compute QED, solubility, and SA scores
            novel_qed = novel_data.apply(QED.qed)
            novel_solubility = novel_data.apply(solubility)
            novel_sa = novel_data.apply(SA)
            novel_drd2 = novel_data.apply(DRD2)

            # Compute means
            novel_mean_qed = novel_qed.mean() if not novel_qed.empty else 0
            novel_mean_solubility = novel_solubility.mean() if not novel_solubility.empty else 0
            novel_mean_sa = novel_sa.mean() if not novel_sa.empty else 0
            novel_mean_drd2 = novel_drd2.mean() if not novel_drd2.empty else 0
        else: 
            novel_mean_qed = 0
            novel_mean_solubility = 0
            novel_mean_sa = 0
            novel_mean_drd2 = 0
            
        print('\nResults Report:')
        print('*'*80)
        print("Total Mols:   {}".format(len(generated_mols)))
        print("Validity:     {}    ({:.2f}%)".format(len(valid_smiles), validity*100))
        print("Uniqueness:   {}    ({:.2f}%)".format(len(unique_smiles), uniqueness*100))
        print("Novelty:      {}    ({:.2f}%)".format(len(novel_smiles), novelty*100))
        print("Total:               {:.2f}%".format(total*100))
        print("Diversity:           {:.2f}".format(diversity))
        print("Novel Mean QED:           {:.2f}".format(novel_mean_qed))
        print("Novel Mean Solubility:           {:.2f}".format(novel_mean_solubility))
        print("Novel Mean SA:           {:.2f}".format(novel_mean_sa))
        print("Novel Mean DRD2:           {:.2f}".format(novel_mean_drd2))
        print('\n')
        print('Samples of Novel SMILES:')


        logging.info('\nResults Report:')
        logging.info('*'*80)
        logging.info("Total Mols:   {}".format(len(generated_mols)))
        logging.info("Validity:     {}    ({:.2f}%)".format(len(valid_smiles), validity*100))
        logging.info("Uniqueness:   {}    ({:.2f}%)".format(len(unique_smiles), uniqueness*100))
        logging.info("Novelty:      {}    ({:.2f}%)".format(len(novel_smiles), novelty*100))
        logging.info("Total:               {:.2f}%".format(total*100))
        logging.info("Diversity:           {:.2f}".format(diversity))
        logging.info("Novel Mean QED:           {:.2f}".format(novel_mean_qed))
        logging.info("Novel Mean Solubility:           {:.2f}".format(novel_mean_solubility))
        logging.info("Novel Mean SA:           {:.2f}".format(novel_mean_sa))
        logging.info("Novel Mean DRD2:           {:.2f}".format(novel_mean_drd2))
        logging.info('\n')
        logging.info('Samples of Novel SMILES:')

        
        if len(novel_smiles) >= 5:
            for i in range(5):
                print(novel_smiles[i])
                logging.info(novel_smiles[i])
        else:
            for i in range(len(novel_smiles)):
                print(novel_smiles[i])
                logging.info(novel_smiles[i])
        print('\n')  
        logging.info('\n')  
        
        # Close the file handle
        #logging.shutdown()

    return validity, uniqueness, novelty, total, diversity, novel_mean_qed, novel_mean_solubility, novel_mean_sa, novel_mean_drd2



















































































