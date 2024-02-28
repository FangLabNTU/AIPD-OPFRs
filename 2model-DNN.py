from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import models,layers
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from sklearn.metrics import roc_auc_score, confusion_matrix
import math
import joblib
from sklearn.cluster import KMeans
import datetime
import shap
import os
import cairosvg
import matplotlib.pyplot as plt

file = 'figs'
if not os.path.exists('./'+file):
    os.makedirs('./'+file)
file = 'model'
if not os.path.exists('./'+file):
    os.makedirs('./'+file)

#%% load data
input_data=pd.read_excel("./data/wash_data.xlsx")
nBits=1024
class_train=2
dpi=150
print("Done!")

#%%Get Morgan Fingerprints
df=input_data
df=df.sort_values(by='LOI', ascending=True)
df = df.reset_index(drop=True)
fingerprints=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=nBits) for smi in list(df.iloc[:, 1])]
X = np.array(fingerprints)
y = df['Label'].values

feature_names = ['Morgan_' + str(s) for s in range(0, nBits)] # col name
x_mass=df["molecular_weight"].values
y_LOI=df["LOI"].values

print("Done!")
#%%Cluster,choose class
cluster = KMeans(n_clusters=3, n_init=10,random_state=0)
cluster.fit(X)
x_class=cluster.predict(X)
index_0=np.where(x_class == 0) 
index_1=np.where(x_class == 1) 
index_2=np.where(x_class == 2)  

#%%plot
plt.scatter(x_mass[index_0],y_LOI[index_0], c="b",label="class 0")
plt.scatter(x_mass[index_1],y_LOI[index_1], c="r",label="class 1")
plt.scatter(x_mass[index_2],y_LOI[index_2], c="g",label="class 2")
plt.legend()
plt.xlabel("Molecular weight")
plt.ylabel("LOI")
plt.savefig("./figs/all_cluster.tiff",dpi=dpi)
plt.show()
#%%
plt.close()

#%%Class=2,model
if class_train==0:
    class_other1,class_other2=1,2
elif class_train==1:
    class_other1,class_other2=0,2
elif class_train==2:
    class_other1,class_other2=0,1
exec('index_train = index_{}'.format(class_train))
exec('index_other1 = index_{}'.format(class_other1))
exec('index_other2 = index_{}'.format(class_other2))
index_other = np.concatenate((index_other1[0], index_other2[0])) 

df_other=df.loc[index_other] 
x_other=X[index_other] 
y_other=y[index_other] 
df=df.loc[index_train] 
df.to_excel("./output/data_train.xlsx")
X=X[index_train]  
y=y[index_train]  
print("Done!")

#%%Splitting the training set and the test set
x_train= np.delete(X, np.arange(0, len(X), 5),axis=0)
x_test=X[::5]
y_train=np.delete(y, np.arange(0, len(y), 5),axis=0)
y_test=y[::5]
print("Done!")

#%%Build the DNN model
tf.keras.backend.clear_session() 
model = models.Sequential()
model.add(layers.Dense(20,activation = 'relu',input_shape=(nBits,)))
model.add(layers.Dense(10,activation = 'relu' ))
model.add(layers.Dense(1,activation = 'sigmoid' ))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['AUC'])

model.summary()

#%%train
model_path="./model/best_model.h5"
epochs=500
batch_size=64

#save
cp_callback=tf.keras.callbacks.ModelCheckpoint(filepath= model_path,save_weights_only=False,save_best_only=True)
start_t = datetime.datetime.now() 
history=model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test,y_test),callbacks=[cp_callback])
end_t = datetime.datetime.now() 
elapsed_sec = (end_t - start_t).total_seconds()

model.save(f"./model/last_model-{end_t.year}-{end_t.month}-{end_t.day}-{end_t.hour}-{end_t.minute}.h5") #save model
print("Time: " + "{:.1f}".format(elapsed_sec) + "s")



#%%plot
def plot_metric(history,metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.savefig("./figs/Training and validation loss.tiff",dpi=dpi)
    plt.show()

plot_metric(history,"loss")
best_epochs=history.history['val_loss'].index(min(history.history['val_loss']))
print("best_epochs: ",best_epochs)

#%% load model
model = tf.keras.models.load_model(model_path)
with open (f'result-{end_t.year}-{end_t.month}-{end_t.day}-{end_t.hour}-{end_t.minute}.txt','w') as f:
    print("nBits: ",nBits,file = f)
    print("best_epochs: ",best_epochs,file = f)
    print("class num: ",class_train,file = f)
    print("",file = f)
    print("class 0: ",len(index_0[0]),file = f)
    print("class 1: ",len(index_1[0]),file = f)
    print("class 2: ",len(index_2[0]),file = f)
print("Done!")

#%% Model evaluation
#train
name="train"

exec(f"x_data = x_{name}")
exec(f"y_data = y_{name}")
y_pred = model(x_data)
auc_roc_score = roc_auc_score(y_data, y_pred)
loss=tf.keras.losses.binary_crossentropy(y_data.reshape(1,-1), y_pred.numpy().reshape(1,-1)).numpy()[0]
y_pred = np.where(y_pred < 0.5, 0, 1).tolist()
y_pred_print = [round(y[0], 0) for y in y_pred]

tn, fp, fn, tp = confusion_matrix(y_data, y_pred_print).ravel()
se = tp / (tp + fn)
sp = tn / (tn + fp)  
accuracy = (tp + tn) / (tp + fn + tn + fp)
mcc = (tp * tn - fn * fp) / math.sqrt((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp))
P = tp / (tp + fp)
F1 = (P * se * 2) / (P + se)
BA = (se + sp) / 2

with open (f'result-{end_t.year}-{end_t.month}-{end_t.day}-{end_t.hour}-{end_t.minute}.txt','a') as f:
    print("",file = f)
    print(f"{name}: ",file = f)
    print("loss: ",format(loss, '.3f'),file = f)
    print("accuracy: ",format(accuracy, '.3f'),file = f)
    print("auc_roc_score: ",format(auc_roc_score, '.3f'),file = f)
    print("mcc: ",format(mcc, '.3f'),file = f)
    print("tp: ",tp,file = f)
    print("tn: ",tn,file = f)
    print("fn: ", fn,file = f)
    print("fp: ",fp,file = f)
    print("se: ",format(se, '.3f'),file = f)
    print("sp: ", format(sp, '.3f'),file = f)
    print("F1: ",format(F1, '.3f'),file = f)
    print("BA: ", format(BA, '.3f'),file = f)
print("Done!")

#test
name="test"

exec(f"x_data = x_{name}")
exec(f"y_data = y_{name}")
y_pred = model(x_data)
auc_roc_score = roc_auc_score(y_data, y_pred)
loss=tf.keras.losses.binary_crossentropy(y_data.reshape(1,-1), y_pred.numpy().reshape(1,-1)).numpy()[0]
y_pred = np.where(y_pred < 0.5, 0, 1).tolist()
y_pred_print = [round(y[0], 0) for y in y_pred]

tn, fp, fn, tp = confusion_matrix(y_data, y_pred_print).ravel()
se = tp / (tp + fn)
sp = tn / (tn + fp)  # 也是R
accuracy = (tp + tn) / (tp + fn + tn + fp)
mcc = (tp * tn - fn * fp) / math.sqrt((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp))
P = tp / (tp + fp)
F1 = (P * se * 2) / (P + se)
BA = (se + sp) / 2

with open (f'result-{end_t.year}-{end_t.month}-{end_t.day}-{end_t.hour}-{end_t.minute}.txt','a') as f:
    print("",file = f)
    print(f"{name}: ",file = f)
    print("loss: ",format(loss, '.3f'),file = f)
    print("accuracy: ",format(accuracy, '.3f'),file = f)
    print("auc_roc_score: ",format(auc_roc_score, '.3f'),file = f)
    print("mcc: ",format(mcc, '.3f'),file = f)
    print("tp: ",tp,file = f)
    print("tn: ",tn,file = f)
    print("fn: ", fn,file = f)
    print("fp: ",fp,file = f)
    print("se: ",format(se, '.3f'),file = f)
    print("sp: ", format(sp, '.3f'),file = f)
    print("F1: ",format(F1, '.3f'),file = f)
    print("BA: ", format(BA, '.3f'),file = f)
print("Done!")

#other
name="other"

exec(f"x_data = x_{name}")
exec(f"y_data = y_{name}")
y_pred = model(x_data)
auc_roc_score = roc_auc_score(y_data, y_pred)
loss=tf.keras.losses.binary_crossentropy(y_data.reshape(1,-1), y_pred.numpy().reshape(1,-1)).numpy()[0]
y_pred = np.where(y_pred < 0.5, 0, 1).tolist()
y_pred_print = [round(y[0], 0) for y in y_pred]

tn, fp, fn, tp = confusion_matrix(y_data, y_pred_print).ravel()
se = tp / (tp + fn)
sp = tn / (tn + fp)  
accuracy = (tp + tn) / (tp + fn + tn + fp)
mcc = (tp * tn - fn * fp) / math.sqrt((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp))
P = tp / (tp + fp)
F1 = (P * se * 2) / (P + se)
BA = (se + sp) / 2

with open (f'result-{end_t.year}-{end_t.month}-{end_t.day}-{end_t.hour}-{end_t.minute}.txt','a') as f:
    print("",file = f)
    print(f"{name}: ",file = f)
    print("loss: ",format(loss, '.3f'),file = f)
    print("accuracy: ",format(accuracy, '.3f'),file = f)
    print("auc_roc_score: ",format(auc_roc_score, '.3f'),file = f)
    print("mcc: ",format(mcc, '.3f'),file = f)
    print("tp: ",tp,file = f)
    print("tn: ",tn,file = f)
    print("fn: ", fn,file = f)
    print("fp: ",fp,file = f)
    print("se: ",format(se, '.3f'),file = f)
    print("sp: ", format(sp, '.3f'),file = f)
    print("F1: ",format(F1, '.3f'),file = f)
    print("BA: ", format(BA, '.3f'),file = f)
print("Done!")

#%% plot
x_mass=df["molecular_weight"].values
x_train_mass=np.delete(df["molecular_weight"].values, np.arange(0, len(X), 5),axis=0)
x_test_mass=df["molecular_weight"].values[::5]

y_LOI=df["LOI"].values
y_train_LOI=np.delete(df["LOI"].values, np.arange(0, len(y), 5),axis=0)
y_test_LOI=df["LOI"].values[::5]


plt.scatter(x_train_mass,y_train_LOI, c="b", label='Train')
plt.scatter(x_test_mass, y_test_LOI, c="r", marker='s', label='Test')
plt.legend()
plt.xlabel("Molecular weight")
plt.ylabel("LOI")
plt.savefig("./figs/scatter.tiff",dpi=dpi)
plt.show()
plt.close()
print("Done!")

#%% PCA plot
pca=PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

x_class = pca.transform(X)
x_train_new = pca.transform(x_train)
x_test_new = pca.transform(x_test)

plt.scatter(x_train_new[:, 0],x_train_new[:, 1], c="b", label='Train')
plt.scatter(x_test_new[:, 0], x_test_new[:, 1], c="r", marker='s', label='Test')
plt.legend()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("./figs/pca.tiff",dpi=dpi)
plt.show()
plt.close()
print("Done!")

#%%plot
train_counts = [np.count_nonzero(y_train==0), np.count_nonzero(y_train==1)]
test_counts = [np.count_nonzero(y_test==0), np.count_nonzero(y_test==1)]
other_counts = [np.count_nonzero(y_other==0), np.count_nonzero(y_other==1)]
x_labels = ["0", "1"]
barWidth = 0.25
r1 = np.arange(len(x_labels))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]


plt.bar(r1, train_counts, width=barWidth, edgecolor='white', label='Train')
plt.bar(r2, test_counts, width=barWidth, edgecolor='white', label='Test')
plt.bar(r3, other_counts, width=barWidth, edgecolor='white',label='Other')
plt.xticks([r + barWidth for r in range(len(r1))], x_labels)
plt.legend()

plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Counts of Label in Train and Test Set')
plt.savefig("./figs/count_label.tiff",dpi=dpi)
plt.show()
plt.close()
print("Done!")

