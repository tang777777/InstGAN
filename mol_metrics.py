import os
import gzip
import math
import pickle
import random
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import rdBase
from math import exp, log
from copy import deepcopy
from rdkit.Chem import QED
from rdkit import DataStructs
from rdkit.Chem import PandasTools, Crippen, Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
rdBase.DisableLog('rdApp.error')
# ============================================================================


# ============================================================================
# Select chemical properties
def reward_fn(properties, generated_mols):
    if properties == 'druglikeness':
        vals = batch_druglikeness(generated_mols) 
    elif properties == 'solubility':
        vals = batch_solubility(generated_mols)
    elif properties == 'synthesizability':
        vals = batch_SA(generated_mols)   
    return vals

# Diversity
def batch_diversity(smiles):
    scores = []
    df = pd.DataFrame({'smiles': smiles})
    PandasTools.AddMoleculeColumnToFrame(df, 'smiles', 'mol')
    fps = [GetMorganFingerprintAsBitVect(m, 4, nBits=2048) for m in df['mol'] if m is not None]
    for i in range(1, len(fps)):
        scores.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i], returnDistance=True))
    return np.mean(scores)



# each solubility
def solubility(mol): 
    low_logp = -2.12178879609
    high_logp = 6.0429063424
    logp = Crippen.MolLogP(mol)
    val = (logp - low_logp) / (high_logp - low_logp)
    val = np.clip(val, 0.1, 1.0)
    return val


# Read synthesizability model
def readSAModel(filename='SA_score.pkl.gz'):
    model_data = pickle.load(gzip.open(filename))
    outDict = {}
    for i in model_data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    SA_model = outDict
    return SA_model
SA_model = readSAModel()


# each synthesizability
def SA(mol):
    if mol is not None and mol.GetNumAtoms() > 1:
        # fragment score
        fp = Chem.AllChem.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        score1 = 0.
        nf = 0
        for bitId, v in fps.items():
            nf += v
            sfp = bitId
            score1 += SA_model.get(sfp, -4) * v
        score1 /= nf
        # features score
        nAtoms = mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        nSpiro = Chem.AllChem.CalcNumSpiroAtoms(mol)
        nBridgeheads = Chem.AllChem.CalcNumBridgeheadAtoms(mol)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1
        sizePenalty = nAtoms**1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)
        score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty
        score3 = 0.
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * .5
        sascore = score1 + score2 + score3
        min = -4.0
        max = 2.5
        sascore = 11. - (sascore - min + 1) / (max - min) * 9.
        # smooth the 10-end
        if sascore > 8.:
            sascore = 8. + math.log(sascore + 1. - 9.)
        if sascore > 10.:
            sascore = 10.0
        elif sascore < 1.:
            sascore = 1.0
        val = (sascore - 5) / (1.5 - 5)
        val = np.clip(val, 0.1, 1.0)
    else:
        val = 0.0
    return val

# Read DRD2 model
DRD2_model = pickle.load(open('DRD2_score.sav', 'rb'))

def DRD2(mol):
    try:
        morgan = [GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        val = DRD2_model.predict_proba(np.array(morgan))[:, 1]
        val = val[0]


    except ValueError:
        val = 0.0
    return val