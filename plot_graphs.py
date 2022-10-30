# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


# PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
import pdb
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

from utils import (
    preprocess_digits,
    train_dev_test_split,
    h_param_tuning,
    data_viz,
    pred_image_viz,
    get_all_h_param_comb,
    tune_and_save,
)
from joblib import dump, load

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

# hyperparamter for decision tree.
min_samples_split_list = [2,3,4,5,6,7]
max_depth_list = [1,2,3,4,5,6]


params = {}
params["gamma"] = gamma_list
params["C"] = c_list

decision_params = {}
decision_params["min_samples_split"] = min_samples_split_list
decision_params["max_depth"] = max_depth_list

h_param_comb = get_all_h_param_comb(params)
decision_h_param_comb = get_all_h_param_comb(decision_params)

# print(decision_h_param_comb)

# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits


x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
    data, label, train_frac, dev_frac
)

# PART: Define the model
# Create a classifier: a support vector classifier
clf = svm.SVC()
decision = DecisionTreeClassifier()

# define the evaluation metric
metric = metrics.accuracy_score


actual_model_path = tune_and_save(
    clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path=None
)

decision_actual_model_path = tune_and_save(
    decision, x_train, y_train, x_dev, y_dev, metric, decision_h_param_comb, model_path=None
)

performance = pd.DataFrame()

# 2. load the best_model
best_model = load(actual_model_path)
decison_best_model = load(decision_actual_model_path)

# PART: Get test set predictions
# Predict the value of the digit on the test subset
predicted = best_model.predict(x_test)
decision_predicted = decison_best_model.predict(x_test)

performance.loc[0,'Type'] = "svc"
performance.loc[1,'Type'] = "decision"

performance.loc[0,'split'] = 1
performance.loc[1,'split'] = 1

performance.loc[0,'accuracy_train'] = metric(y_train,best_model.predict(x_train))
performance.loc[1,'accuracy_train'] = metric(y_train,decison_best_model.predict(x_train))

performance.loc[0,'accuracy_test'] = metric(y_test,best_model.predict(x_test))
performance.loc[1,'accuracy_test'] = metric(y_test,decison_best_model.predict(x_test))

performance.loc[0,'accuracy_dev'] = metric(y_dev,best_model.predict(x_dev))
performance.loc[1,'accuracy_dev'] = metric(y_dev,decison_best_model.predict(x_dev))

pred_image_viz(x_test, predicted)
pred_image_viz(x_test,decision_predicted)

temp = 2
for i in range(1,5):
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
    data, label, train_frac, dev_frac
    )

    performance.loc[temp,'Type'] = "svc"
    performance.loc[temp,'split'] = i+1
    train_pred_svc = best_model.predict(x_train)
    test_pred_svc = best_model.predict(x_test)
    dev_pred_svc = best_model.predict(x_dev)
    performance.loc[temp,'accuracy_train'] = metric(y_train,train_pred_svc)
    performance.loc[temp,'accuracy_test'] = metric(y_test,test_pred_svc)
    performance.loc[temp,'accuracy_dev'] = metric(y_dev,dev_pred_svc)

    da = ["train","test","dev"]
    actual_data_svc = [y_train,y_test,y_dev]
    for j,data_ in enumerate([train_pred_svc,test_pred_svc,dev_pred_svc]):
        unique, count = np.unique(data_, return_counts = True)
        unique_actual, count_actual = np.unique(actual_data_svc[j], return_counts = True)
        print(f"Predicted Label for svc ({da[j]}) {list(zip(unique,count))}")
        print(f"actual Label for svc    ({da[j]}) {list(zip(unique_actual,count_actual))}")

    performance.loc[temp+1,'Type'] = "decision"
    performance.loc[temp+1,'split'] = i+1
    train_pred_dec = decison_best_model.predict(x_train)
    test_pred_dec = decison_best_model.predict(x_test)
    dev_pred_dec = decison_best_model.predict(x_dev)
    performance.loc[temp+1,'accuracy_train'] = metric(y_train,train_pred_dec)
    performance.loc[temp+1,'accuracy_test'] = metric(y_test,test_pred_dec)
    performance.loc[temp+1,'accuracy_dev'] = metric(y_dev,dev_pred_dec)
    
    for j,data_ in enumerate([train_pred_dec,test_pred_dec,dev_pred_dec]):
        unique, count = np.unique(data_, return_counts = True)
        unique_actual, count_actual = np.unique(actual_data_svc[j], return_counts = True)
        print(f"Predicted Label for decision ({da[j]}) {list(zip(unique,count))}")
        print(f"actual Label for decision    ({da[j]}) {list(zip(unique_actual,count_actual))}")

    temp += 2



# 4. report the test set accurancy with that best model.
# PART: Compute evaluation metrics
# print(
#     f"Classification report for classifier {best_model}:\n"
#     f"{metrics.classification_report(y_test, predicted)}\n"
# )

# print(
#     f"Classification report for classifier {decison_best_model}:\n"
#     f"{metrics.classification_report(y_test, decision_predicted)}\n"
# )

# mean and standard daviation of model performance.

# svc
svc_perf = performance[performance['Type']=="svc"]
print(f"Mean of SVC (Train) performance : {svc_perf['accuracy_train'].mean()}")
print(f"Mean of SVC (Test) performance : {svc_perf['accuracy_test'].mean()}")
print(f"Mean of SVC (Dev) performance : {svc_perf['accuracy_dev'].mean()}")

print(f"Sandard Daviation of SVC (Train) performance : {svc_perf['accuracy_train'].std()}")
print(f"Sandard Daviation of SVC (Test) performance : {svc_perf['accuracy_test'].std()}")
print(f"Sandard Daviation of SVC (Dev) performance : {svc_perf['accuracy_dev'].std()}")

#decison tree
decision_perf = performance[performance['Type']=="decision"]
print(f"Mean of Decision (Train) performance : {decision_perf['accuracy_train'].mean()}")
print(f"Mean of Decision (Test) performance : {decision_perf['accuracy_test'].mean()}")
print(f"Mean of Decision (Dev) performance : {decision_perf['accuracy_dev'].mean()}")

print(f"Sandard Daviation of Decision (Train) performance : {decision_perf['accuracy_train'].std()}")
print(f"Sandard Daviation of Decision (Test) performance : {decision_perf['accuracy_test'].std()}")
print(f"Sandard Daviation of Decision (Dev) performance : {decision_perf['accuracy_dev'].std()}")

#writing the data to file
write_data = ["\n run svm decision_tree"]
for i in range(1,6):
    split_data = performance[performance['split']==i]
    acc_svc = split_data[split_data['Type']=='svc']['accuracy_test'].values
    acc_decison = split_data[split_data['Type']=='decision']['accuracy_test'].values
    ta = "\n" +str(i)+" "+str(acc_svc)+" "+str(acc_decison)
    write_data.append(ta)

write_data.append("\n mean"+" "+str(svc_perf['accuracy_test'].mean())+" "+str(decision_perf['accuracy_test'].mean()))
write_data.append("\n std" +" "+str(svc_perf['accuracy_test'].std())+" "+str(decision_perf['accuracy_test'].std()))

with open("readme.md","a") as f:
    f.writelines(write_data)
