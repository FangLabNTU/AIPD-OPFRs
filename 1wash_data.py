import pandas as pd
from rdkit import Chem
from standardiser import standardise
import logging
from rdkit.Chem import Descriptors
import rdkit

input_data = pd.read_excel('./data/input.xlsx')

#%%wash dataset
df=input_data
df=df.dropna()

for i in df.index:
    try:
        smi = df.loc[i, 'SMILES']
        print(smi)
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        parent = standardise.run(mol)
        mol_ok_smi = Chem.MolToSmiles(parent)
        df.loc[i, 'SMILES'] = mol_ok_smi
        print(i, 'done')
    except standardise.StandardiseException as e:
        logging.warning(e.message)
df.drop_duplicates(keep='first', inplace=True)


#%%
molweight = []
for smi in list(df['SMILES']):
    molweight.append(Descriptors.MolWt(Chem.MolFromSmiles(smi)))
print(molweight)
df['molecular_weight'] = molweight

logP = []
for smi in list(df['SMILES']):
    logP.append(Descriptors.MolLogP(Chem.MolFromSmiles(smi)))
print(logP)
df['logP'] = logP
#df = df[df['molecular_weight']<=1000]

#%% classification
df["Label"]=0
df.loc[df["LOI"]>30.1,"Label"]=1
#df[df["Label"]==1].shape

#%%save
df.to_excel('./data/wash_data.xlsx',index=None)

