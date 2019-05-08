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
from sklearn.model_selection import GridSearchCV

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
from collections import Counter
from keras.wrappers.scikit_learn import KerasClassifier
import keras



A = open('TCGA_new_pre_second.pckl', 'rb')
[dropped_genes_final, dropped_gene_name, dropped_Ens_id, samp_id_new, diag_name_new,
 project_ids_new] = pickle.load(A)
A.close()

f = open('TCGA_new_pre_first.pckl', 'rb')
[_, _, _, _, remain_cancer_ids_ind, remain_normal_ids_ind] = pickle.load(f)
f.close()


## embedding labels
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(project_ids_new)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

X_cancer_samples = dropped_genes_final.iloc[:, remain_cancer_ids_ind].T.values
X_normal_samples = dropped_genes_final.iloc[:, remain_normal_ids_ind].T.values
onehot_encoded_cancer_samples = onehot_encoded[remain_cancer_ids_ind]
onehot_encoded_normal_samples = onehot_encoded[remain_normal_ids_ind]

X_cancer_samples_mat = np.concatenate((X_cancer_samples,np.zeros((len(X_cancer_samples),9))),axis=1)
## add nine zeros to the end of each sample
# This line dimension needs to be changed for different kernel sizes
X_cancer_samples_mat = np.reshape(X_cancer_samples_mat, (-1, 71, 100))

## This line is useful when only one fold training is needed
x_train, x_test, y_train, y_test = train_test_split(X_cancer_samples_mat, onehot_encoded_cancer_samples,
                                                    stratify=onehot_encoded_cancer_samples,
                                                    test_size=0.25, random_state=42)
# adding one dimention to feed into CNN
img_rows, img_cols = len(x_test[0]), len(x_test[0][0])

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
# X_normal_samples = X_normal_samples.reshape(X_normal_samples.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


## Model

def make_model(dense_layer_sizes, filters, kernel_size):
    img_rows, img_cols = len(x_test[0]), len(x_test[0][0])
    num_classes = len(y_train[0])
    input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    ## *********** First layer Conv
    model.add(Conv2D(filters, kernel_size=kernel_size, strides=stride,
                     input_shape=input_shape))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(LeakyReLU())
    model.add(MaxPooling2D(2, 2))
    model.output_shape

    ## ********* Classification layer
    model.add(Flatten())
    model.add(Dense(dense_layer_sizes, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))
    model.output_shape

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])
    model.summary()
    return model
dense_size_candidates = [64, 128, 512]
my_classifier = KerasClassifier(make_model, batch_size=128)
validator = GridSearchCV(my_classifier,
                         param_grid={'dense_layer_sizes': dense_size_candidates,
                                     # epochs is avail for tuning even when not
                                     # an argument to model building function
                                     'epochs': [25],
                                     'filters': [8, 16, 32, 64], 'stride': [(1, 1),(2, 2),(5, 5)],
                                     'kernel_size': [(7, 7), (10, 10), (15, 15), (20, 20)]},
                         scoring='neg_log_loss',
                         n_jobs=1)
validator.fit(x_train, y_train)
import csv
print('The parameters of the best model are: ')
print(validator.best_params_)
# write it in a excel file
with open('results_runs50.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in validator.cv_results_.items():
       writer.writerow([key, value])
# validator.best_estimator_ returns sklearn-wrapped version of best model.
# validator.best_estimator_.model returns the (unwrapped) keras model
best_model = validator.best_estimator_.model
metric_names = best_model.metrics_names
metric_values = best_model.evaluate(x_test, y_test)
for metric, value in zip(metric_names, metric_values):
    print(metric, ': ', value)
