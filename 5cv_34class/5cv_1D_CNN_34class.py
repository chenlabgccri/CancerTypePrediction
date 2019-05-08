'''
This code is written by Milad Mostavi, one of authors of
"Convolutional neural network models for cancer type prediction based on gene expression" paper.
Please cite this paper in the case it was useful in your research
'''
import pickle
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
from sklearn.model_selection import StratifiedKFold
from collections import Counter







A = open('TCGA_new_pre_second.pckl', 'rb')
[dropped_genes_final, dropped_gene_name, dropped_Ens_id, samp_id_new, diag_name_new,
 project_ids_new] = pickle.load(A)
A.close()

f = open('TCGA_new_pre_first.pckl', 'rb')
[_, _, _, _, remain_cancer_ids_ind, remain_normal_ids_ind] = pickle.load(f)
f.close()

batch_size = 128
epochs = 50
seed = 7
np.random.seed(seed)


X_cancer_samples =dropped_genes_final.iloc[:,remain_cancer_ids_ind].T.values
X_normal_samples = dropped_genes_final.iloc[:,remain_normal_ids_ind].T.values

name_cancer_samples = project_ids_new[remain_cancer_ids_ind]
name_normal_samples = ['Normal Samples'] *len(X_normal_samples)

X_cancer_samples_34 = np.concatenate((X_cancer_samples,X_normal_samples))
X_names = np.concatenate((name_cancer_samples,name_normal_samples))
X_cancer_samples_mat = np.concatenate((X_cancer_samples_34,np.zeros((len(X_cancer_samples_34),9))),axis=1)
X_cancer_samples_mat = np.reshape(X_cancer_samples_mat, (-1, 71, 100))




kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
cv_yscores = []
Y_test =[]

input_Xs = X_cancer_samples_mat
y_s = X_names

img_rows, img_cols = len(input_Xs[0][0]), len(input_Xs[0])
num_classes = len(set(y_s))

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_s)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

i = 0
for train, test in kfold.split(X_cancer_samples_34, y_s):   # input_Xs in normal case and shuffled should be shuffled_Xs

    input_Xs = input_Xs.reshape(input_Xs.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    input_Xs = input_Xs.astype('float32')

    num_classes = len(onehot_encoded[0])

    model = Sequential()
    ## *********** First layer Conv
    model.add(Conv2D(32, kernel_size=(1, 71), strides=(1, 1),
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(1, 2))
    ## ********* Classification layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])
    callbacks = [EarlyStopping(monitor='categorical_accuracy', patience=3, verbose=0)]
    if i==0:
        model.summary()
        i = i +1
    history = model.fit(input_Xs[train], onehot_encoded[train],
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0, callbacks=callbacks, validation_data=(input_Xs[test], onehot_encoded[test]))
    scores = model.evaluate(input_Xs[test], onehot_encoded[test], verbose=0)
    y_score = model.predict(input_Xs[test])
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    cvscores.append(scores[1] * 100)
    Y_test.append(onehot_encoded[test])
    cv_yscores.append(y_score)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
cv_yscores = np.concatenate(cv_yscores)
Y_test = np.concatenate(Y_test)






