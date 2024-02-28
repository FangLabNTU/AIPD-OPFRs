# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import datetime
from tensorflow.keras import models,layers
print("Done!")
start_t = datetime.datetime.now()

#%% load model
input_data=pd.read_excel("./data/input.xlsx")
nBits=1024
model_path="output/model/DNN-best-model.h5"
print("Done!")

tf.keras.backend.clear_session()
model=models.Sequential()
model.add(layers.Dense(20,activation="relu",input_shape=(nBits,)))
model.add(layers.Dense(10,activation="relu"))
model.add(layers.Dense(1,activation="sigmoid"))
model.summary()

#load model
model.load_weights(model_path)
print("Done!")

#%%Get Morgan Fingerprints
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
print("Done!")

#%%
x_validation = np.array(fingerprints)
df.iloc[index_list]
df.iloc[fail_index_list]
print("Done!")

#%%Pred
name="validation"
exec(f"x_data = x_{name}")

y_pred = model(x_data)
y_pred = np.where(y_pred < 0.5, 0, 1).tolist()
y_pred_print = [round(y[0], 0) for y in y_pred]
df.loc[index_list,"Label"]=y_pred_print
save_data=df
print("Done!")

#%%Save Morgan Fingerprints
list=[507,393,114,563,179,887,841,190,794,67]
new_names=["DNN1024_"+str(i) for i in list]
df=pd.DataFrame(x_data)
df=df[list]
df.columns=new_names

save_data=pd.concat([save_data,df],axis=1)
end_t = datetime.datetime.now() 
save_data.to_excel(f"./output/{end_t.date()}.xlsx")
t=end_t-start_t
print("Time: ",t)
print("Finish!")
