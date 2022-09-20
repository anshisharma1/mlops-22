import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.transform import rescale, resize
import numpy as np
import pandas as pd

def train_model_with_hyperparam(digits,target_value):
    
    # defining the scope of hyperparameters.
    gamma_list = [0.01,0.005,0.001,0.0005]
    c_list = [0.1,0.2,0.5,0.7,1,2,5]

    # defining the fraction of the dataset.
    train_frac = 0.8
    test_frac = 0.1
    dev_frac = 0.1

    #PART: data pre-processing -- to remove some noise, to normalize data, format the data to be consumed by mode
    # flatten the images
    n_samples = len(digits)
    data = digits.reshape((n_samples, -1))


    # Splitting the data into train,dev and test set.
    dev_test_frac = 1-train_frac
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(data, target_value, test_size=dev_test_frac, shuffle=True)
    X_test, X_dev, y_test, y_dev = train_test_split(X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True)


    #PART: Define the model
    # Create a classifier: a support vector classifier
    accuracy_score_model = 0
    i=0
    c_best = None
    gamma_best = None
    best_model = None
    metrics_df = pd.DataFrame(columns=['hyperparameter','train','dev','test'])
    for gamma in gamma_list:
        for c in c_list:
            clf = svm.SVC(C=c,gamma=gamma)
            clf.fit(X_train,y_train)
            predicted = clf.predict(X_dev)
            accuracy_score_dev = accuracy_score(y_dev,predicted)
            metrics_df.loc[i,'hyperparameter'] = f'C :{c}, Gamma: {gamma}'
            metrics_df.loc[i,'train'] = accuracy_score(y_train,clf.predict(X_train))
            metrics_df.loc[i,'dev'] = accuracy_score(y_dev,clf.predict(X_dev))
            metrics_df.loc[i,'test'] = accuracy_score(y_test,clf.predict(X_test))
            i += 1
            if accuracy_score_model < accuracy_score_dev:
                accuracy_score_model = accuracy_score_dev
                best_model = clf
                c_best = c
                gamma_best = gamma

    metrics_df.set_index('hyperparameter',inplace=True)
    print(metrics_df)
    print('')
    print(f'Best Hyperparams : C : {c}, gamma : {gamma}')
    print(f'Accuracy of Training set with the best Model: {accuracy_score(y_train,best_model.predict(X_train))}')
    print(f'Accuracy of Dev set with the best Model: {accuracy_score(y_dev,best_model.predict(X_dev))}')
    print(f'Accuracy of Test set with the best Model: {accuracy_score(y_test,best_model.predict(X_test))}')

if __name__ == "__main__":
    #PART: load dataset -- data from csv, tsv, jsonl, pickle
    digits = datasets.load_digits()

    # setting the width and height of the image
    x=8
    y=8
    data = []
    for image_data in digits.images:
        resized_img = resize(image_data, (x, y),anti_aliasing=True)
        data.append(resized_img)

    temp_array = np.array(data)
    print('')
    print(f"Size of the Original Image in Dataset: {digits.images[0].shape}")
    if not (x==8 and y==8):
        print(f"Size of the transformed Image in Dataset: {temp_array[0].shape}")
    print('')

    train_model_with_hyperparam(temp_array, digits.target)


