#%%
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix
import math
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import xgboost
import datetime
print("Done!")

#%% load data
input_data=pd.read_excel("./data/wash_data.xlsx")
nBits=2048

print("Done!")

#%% Get Morgan Fingerprints
df=input_data
df=df.sort_values(by='LOI', ascending=True)
df = df.reset_index(drop=True)
fingerprints=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=nBits) for smi in list(df.iloc[:, 1])]
X = np.array(fingerprints)
y = df['Label'].values
feature_names = ['Morgan_' + str(s) for s in range(0, nBits)] =

x_mass=df["molecular_weight"].values
y_LOI=df["LOI"].values

print("Done!")

#%% Cluster
cluster = KMeans(n_clusters=3, random_state=0)
cluster.fit(X)
x_class=cluster.predict(X)

index_0=np.where(x_class == 0) 
index_1=np.where(x_class == 1) 
index_2=np.where(x_class == 2)  

plt.scatter(x_mass[index_0],y_LOI[index_0], c="b",label="class 0")
plt.scatter(x_mass[index_1],y_LOI[index_1], c="r",label="class 1")
plt.scatter(x_mass[index_2],y_LOI[index_2], c="g",label="class 2")
plt.legend()
plt.xlabel("Molecular weight")
plt.ylabel("LOI")
plt.savefig("all_cluster.tiff",dpi=300)
plt.show()

#%%Class 2 model
index_train=index_1
index_other = np.concatenate((index_2[0], index_0[0])) 

plt.close()
df_other=df.loc[index_other] 
x_other=X[index_other] 
y_other=y[index_other] 
df=df.loc[index_train] 
df.to_excel("data_train.xlsx")
X=X[index_train]  
y=y[index_train] 

#%%Splitting the training set and the test set
X_train= np.delete(X, np.arange(0, len(X), 5),axis=0)
X_test=X[::5]
y_train=np.delete(y, np.arange(0, len(y), 5),axis=0)
y_test=y[::5]
print("Done!")

#%%build model
model = xgboost.XGBClassifier()
param={'max_depth':range(0,20),'min_child_weight':range(0,20)}

gc_model = GridSearchCV(model, param_grid=param, cv=5, scoring='roc_auc') 
start_t = datetime.datetime.now()
gc_model.fit(X_train, y_train) 
end_t = datetime.datetime.now() 
elapsed_sec = (end_t - start_t).total_seconds() 
joblib.dump(gc_model,f'gc_model-{end_t.year}-{end_t.month}-{end_t.day}-{end_t.hour}-{end_t.minute}.pkl') #save model
print("best params: ",gc_model.best_params_)
print("Time: " + "{:.2f}".format(elapsed_sec) + " 秒")

#%% Test
y_pred = gc_model.predict_proba(X_test)[:, 1]
auc_roc_score = roc_auc_score(y_test, y_pred)
y_pred_print = [round(y, 0) for y in y_pred]
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_print).ravel()
se = tp / (tp + fn)
sp = tn / (tn + fp)  
accuracy = (tp + tn) / (tp + fn + tn + fp)
mcc = (tp * tn - fn * fp) / math.sqrt((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp))
P = tp / (tp + fp)
F1 = (P * se * 2) / (P + se)
BA = (se + sp) / 2

with open (f'result-{end_t.year}-{end_t.month}-{end_t.day}-{end_t.hour}-{end_t.minute}.txt','w') as f:
    print("best params: ",gc_model.best_params_,file = f)
    print("tp: ",tp,file = f)
    print("tn: ",tn,file = f)
    print("fn: ", fn,file = f)
    print("fp: ",fp,file = f)
    print("se: ",format(se, '.3f'),file = f)
    print("sp: ", format(sp, '.3f'),file = f)
    print("mcc: ",format(mcc, '.3f'),file = f)
    print("accuracy: ",format(accuracy, '.3f'),file = f)
    print("auc_roc_score",format(auc_roc_score, '.3f'),file = f)
    print("F1: ",format(F1, '.3f'),file = f)
    print("BA: ", format(BA, '.3f'),file = f)
    print("class train: ",len(index_train[0]),file = f)
    print("class 0: ",len(index_0[0]),file = f)
    print("class 1: ",len(index_1[0]),file = f)
    print("class 2: ",len(index_2[0]),file = f)
print("Done!")
