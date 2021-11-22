# tabular-playground-series

!pip install -U lightautoml

import pandas as pd
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

train_data = pd.read_csv('../input/tabular-playground-series-nov-2021/train.csv')
test_data = pd.read_csv('../input/tabular-playground-series-nov-2021/test.csv')
sub = pd.read_csv('../input/tabular-playground-series-nov-2021/sample_submission.csv')
train_data.shape, test_data.shape, sub.shape

%%writefile nn_code.py

# Ref.: https://www.kaggle.com/chaudharypriyanshu/tps-nov-nn-starter-keras
# ================================================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from keras.backend import sigmoid
from sklearn.metrics import roc_auc_score
import joblib

train_data = pd.read_csv('../input/tabular-playground-series-nov-2021/train.csv')
test_data = pd.read_csv('../input/tabular-playground-series-nov-2021/test.csv')

#https://bignerdranch.com/blog/implementing-swish-activation-function-in-keras/
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

get_custom_objects().update({'swish': Activation(swish)})

def base_model(hidden_units):
    num_input = keras.Input(shape=(100,), name='num_data')#input layer
    out = keras.layers.Concatenate()([num_input])
    for n_hidden in hidden_units:
        out = keras.layers.Dense(n_hidden, activation='swish')(out)
        out = keras.layers.Dropout(0.3)(out)
    out = keras.layers.Dense(1, activation='sigmoid', name='prediction')(out)
    model = keras.Model(inputs = [num_input],outputs = out)
    return model

def create_nn_preds(train_data, test_data, n_folds = 3):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2020)

    X = train_data.drop(['id', 'target'], axis = 1).values
    y = train_data['target'].values
    X_test = test_data.drop(['id'], axis = 1).values

    nn_oof_pred = np.array([0.0] * len(train_data))
    nn_test_pred = np.array([0.0] * len(test_data))
    es = keras.callbacks.EarlyStopping(
        monitor='val_auc', patience=20, verbose=0,
        mode='max',restore_best_weights=True)

    plateau = keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc', factor=0.2, patience=7, verbose=0,
        mode='max')
    
    hidden_units = (128, 128, 128, 64) 

    for fold, (trn_ind, val_ind) in enumerate(skf.split(y, y)):
        print(f'Training fold {fold + 1}')
        X_train, X_val = X[trn_ind, :], X[val_ind, :]
        y_train, y_val = y[trn_ind], y[val_ind]
        print('CV {}/{}'.format(fold + 1, n_folds)) 
        
        model = base_model(hidden_units)
        model.compile(
            keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics = ['AUC']
        )

        scaler = MinMaxScaler(feature_range=(0, 1))         
        X_tr = scaler.fit_transform(X_train)    
        X_v = scaler.transform(X_val)
        X_t = scaler.transform(X_test)

        model.fit(X_tr, 
                  y_train,               
                  batch_size=2048,
                  epochs=1000,
                  validation_data=(X_v, y_val),
                  callbacks=[es, plateau],
                  validation_batch_size = 2048,
                  shuffle=True,
                  verbose = 1)

        preds = model.predict(X_v).reshape(-1, 1)[:, 0]
        nn_oof_pred[val_ind] = preds
        score = roc_auc_score(y_val, preds)
        print('Fold {}: {:.7f}'.format(fold + 1, score))

        nn_test_pred += model.predict(X_t).reshape(-1, 1)[:, 0] / n_folds     

    K.clear_session()
    return nn_oof_pred, nn_test_pred

nn_oof_pred, nn_test_pred = create_nn_preds(train_data, test_data, 5)
joblib.dump((nn_oof_pred, nn_test_pred), 'nn_preds.pkl')

!python nn_code.py

# Ref.: https://www.kaggle.com/ambrosm/tpsnov21-001-support-vector-classification
# ================================================================================
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

auc_list = []
svm_oof_pred = np.array([0.0] * len(train_data))
svm_test_pred = np.array([0.0] * len(test_data))

N_folds = 10
kf = StratifiedKFold(n_splits=N_folds, shuffle=True, random_state=13)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_data, train_data.target)):
    print(f"Fold {fold}")
    X_tr = train_data.iloc[train_idx]
    X_va = train_data.iloc[val_idx]
    y_tr = X_tr.target
    y_va = X_va.target
    X_tr = X_tr.drop(columns=['id', 'target'])
    X_va = X_va.drop(columns=['id', 'target'])

    # Train
    model = make_pipeline(StandardScaler(), LinearSVC(tol=1e-7, penalty='l2', dual=False, max_iter=2000))
    model.fit(X_tr, y_tr)
    # Validate
    y_pred = model.decision_function(X_va)
    svm_oof_pred[val_idx] = y_pred
    score = roc_auc_score(y_va, y_pred)
    print(score)
    auc_list.append(score)
    
    # Predict for the submission
    svm_test_pred += model.decision_function(test_data.drop(columns=['id'])) / N_folds

avg_auc = sum(auc_list) / len(auc_list)
print(f"Average AUC: {avg_auc:.5f}")

import joblib
nn_oof_pred, nn_test_pred = joblib.load('nn_preds.pkl')

train_data['NN_pred'] = nn_oof_pred
test_data['NN_pred'] = nn_test_pred

train_data['SVM_pred'] = svm_oof_pred
test_data['SVM_pred'] = svm_test_pred

for data in [train_data, test_data]:
    data['SVM_mul_NN'] = data['SVM_pred'] * data['NN_pred']
    data['SVM_div_NN'] = data['SVM_pred'] / (data['NN_pred'] + 1e-6)
    
print('OOF score NN: {:.7f}'.format(roc_auc_score(train_data['target'], train_data['NN_pred'])))
print('OOF score SVM: {:.7f}'.format(roc_auc_score(train_data['target'], train_data['SVM_pred'])))
print('OOF score MUL: {:.7f}'.format(roc_auc_score(train_data['target'], train_data['SVM_mul_NN'])))
print('OOF score DIV: {:.7f}'.format(roc_auc_score(train_data['target'], train_data['SVM_div_NN'])))

task = Task('binary')
automl = TabularAutoML(task = task, timeout = 8 * 3600, cpu_limit = 4, 
                       general_params = {'use_algos': [['cb']]}, 
                       selection_params = {'mode': 0})
oof_pred = automl.fit_predict(train_data, roles = {'target': 'target', 'drop': ['id']}, verbose = 2)
sub['target'] = automl.predict(test_data).data[:, 0]

print('OOF score LightAutoML: {:.7f}'.format(roc_auc_score(train_data['target'], oof_pred.data[:, 0])))

sub.to_csv('submission.csv', index = False)
