# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
import xgboost
import datetime
print("Done!")
start_t=datetime.datetime.now()
#%% load data
input_data=pd.read_excel("./data/input.xlsx")
nBits=2048
model_path="./model/XGB-best-model.pkl"
print("Done!")


#%% load model
model=joblib.load(model_path)
print("Done!")

#%% Get Morgan Fingerprints
df=input_data
fingerprints=[]
fail_index_list=[]
index_list=[]

for i in df.index:
    try:
        print(i)
        smi=df.loc[i,"SMILES"]
        morgan=AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=nBits)
        fingerprints.append(morgan)
        index_list.append(i)
    except:
        print(i,"fail",smi)
        fail_index_list.append(i)

x_validation = np.array(fingerprints)
df.iloc[index_list]
df.iloc[fail_index_list]
print("Done!")

#%% Pred
name="validation"
exec(f"x_data = x_{name}")
y_pred = model.predict(x_data)
df.loc[index_list,"Label"]=y_pred
save_data=df

#%% Save Morgan Fingerprints
list=[507,114,556,1385,807,80,679,319,938,575]
new_names=["XGB2048_"+str(i) for i in list]
df=pd.DataFrame(x_data)
df=df[list]
df.columns=new_names

save_data=pd.concat([save_data,df],axis=1)
end_t = datetime.datetime.now()
save_data.to_excel(f"./data/output-{end_t.date()}.xlsx")
t=end_t-start_t
print("Time: ",t)
print("Finish!")






