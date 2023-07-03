import pandas as pd
import numpy as np
# import seaborn to visualize data that built on top of matplotlib
import seaborn as sns
# import matplotlib
import matplotlib.pyplot as plt
import datetime

from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

# importing different algorithms to train our data and find better model among all algorithms
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# train_test_split is used to split the data into train and test set of given data
# cross_val_score is used to find the score on given model and KFlod
# KFold is used for defining the no.of folds for Cross Validation
# GridSearchCV for grid searching
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# import pipeline to make machine learning pipeline to overcome data leakage problem
from sklearn.pipeline import Pipeline

# import StandardScaler to Column Standardize the data
# many algorithm assumes data to be Standardized
from sklearn.preprocessing import StandardScaler

# import warnings filter
from warnings import simplefilter

from pathlib import Path, PurePath

import json

# Define the stratified_kfold_score function
def stratified_kfold_scores(clf,X,y,n_fold,seed):
    #X,y = X,y
    strat_kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1score_list = []

    for train_index, test_index in strat_kfold.split(X, y):
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        clf.fit(x_train_fold, y_train_fold)
        y_pred = clf.predict(x_test_fold)

        accuracy_test = accuracy_score(y_test_fold, y_pred)
        precision_test = precision_score(y_test_fold, y_pred)
        recall_test = recall_score(y_test_fold, y_pred)
        f1score_test = f1_score(y_test_fold, y_pred)

        accuracy_list.append(accuracy_test)
        precision_list.append(precision_test)
        recall_list.append(recall_test)
        f1score_list.append(f1score_test)

    accuracy = np.array(accuracy_list).mean()
    precision = np.array(precision_list).mean()
    recall = np.array(recall_list).mean()
    f1 = np.array(f1score_list).mean()

    return np.array([accuracy, precision, recall, f1])

def main():
    # ignore all warnings
    simplefilter(action='ignore', category=Warning)

    # Define parameters
    validation_size = 0.30
    seed = 1234
    num_splits = 10
    # scoring = ['recall','f1']
    scoring = 'f1'
    # scoring = 'recall'
    # scorings = ['accuracy','precision','recall','f1']


    ## Read the csv files
    # get current relative path
    cur_dir = Path('.')

    # get current absolute path
    # cur_dir = Path.cwd()

    # features directory
    data_dir = 'features'

    # results directory
    res_dir = 'rbf_random_tuning_SVM_model_results'
    res_path = cur_dir.joinpath(res_dir)

    # join the directory
    parent_dir = cur_dir.joinpath(data_dir)

    # Define file name to read
    # file_name = '01_tweet_count.csv'

    # # wild card search for csv file paths to read
    filePaths = sorted(list(parent_dir.rglob("*.csv")))
    # print(filePaths[0]) -> features/06_impression_count.csv
    # print(filePaths[0].resolve()) -> /home/username/detect_vul_with_tw_trend/features/06_impression_count.csv
    # print(filePaths[0].with_suffix('').as_posix()) -> features/06_impression_count
    # print(filePaths[0].name) -> 06_impression_count.csv
    # print(filePaths[0].stem) -> 06_impression_count

    #  Import the CVE list with flags from the CSV file
    df_flag = pd.read_csv("CVE_list_with_flagH_20230326.csv", index_col = ['CVE'] )
    df_flag_sorted = df_flag.sort_values('CVE')
    df_flag_sorted.drop('Unnamed: 0', inplace = True, axis = 1)
    df = []

    ## evaluate SVM model for each file
    for i, filepath in enumerate(filePaths):
        msg = None
        # extract filename from filepath
        file_name = filepath.name
        file_stem = filepath.stem
        print('\n'+ file_stem)

        # join paths to the file
        file_path = parent_dir.joinpath(file_name)
        # print(file_name) -> 06_impression_count.csv
        # print(file_path) -> features/06_impression_count.csv

        # Import features from a CSV file
        df.append(pd.read_csv(file_path))
        df[i].rename({"Unnamed: 0" : "CVE"}, inplace = True, axis = 1)
        df[i].set_index('CVE', inplace = True)

        # Filter the features which are NaN# %%
        impute(df[i])
        # print(df[i].head())

        # y = df_flag_sorted['flag_H']
        # features_filterd = select_features(df[i], y)
        # features_filterd

        # if the numbers of features are less than CVEs with flag, drop the CVEs that is not in the features
        if (file_name == '02_retweet_count.csv'):
            df_flag_temp = df_flag_sorted
            df_flag_sorted = df_flag_sorted[df_flag_sorted.index.isin(df[i].index)]

        # split data into train and test data
        train_data = df[i]
        target_data = df_flag_sorted['flag_H']

        # .values returns numpy.ndarray
        # using array instead of dataframe to train model because
        # array is faster to compute instead of dataframe
        X = train_data.values
        y = target_data.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = validation_size, random_state = seed)

        # Set the parameters for SVM
        c_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 3, 5, 10, 50, 100, 500, 1000]
        gamma_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 1, 3, 5, 10, 50, 100, 500, 1000]
        # kernel_values = ['rbf']
        # kernel_values = ['rbf', 'sigmoid']
        # kernel_values = ['rbf', 'sigmoid', 'poly']
        # kernel_values = ['linear', 'rbf', 'sigmoid', 'poly']

        param_grid = dict(C=c_values, gamma=gamma_values)
        # param_grid = dict(C=c_values, gamma=gamma_values, kernel=kernel_values)
        # print(param_grid)

        # Create an SVM instance
        clf = SVC(kernel = 'rbf')
        # clf = SVC()


        # StratifiedKFoldの設定
        # initializing kfold by n_splits=10(no.of K)
        skf = StratifiedKFold(n_splits = num_splits, shuffle = True, random_state = seed)
        random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, n_iter = 100, cv=skf, scoring=scoring, verbose=5, n_jobs=-1)
        # results = stratified_kfold_scores(clf,X,y,num_splits,seed)

        random_result = random_search.fit(X_train, y_train)
        print("random_serach.fit ended")

        # fit結果をf1_scoreの降順でCSV保存
        df_cv_results = pd.DataFrame(random_result.cv_results_).sort_values('mean_test_score', ascending = False)
        df_cv_results.to_csv(f"{res_path}/{file_stem}_rbf_random_cv_result.csv")

        # with open (f'{res_path}/{file_stem}_grid_result.json','w') as f:
        #     json.dump(grid_result.cv_results_, f, indent=4)
        print("cv_results_ record ended")

        best_model = random_result.best_estimator_
        pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        f1 = f1_score (y_test, pred)
        recall = recall_score(y_test, pred)
        cf_matrix = confusion_matrix(y_test, pred)

        msg1 = "best_score:{:.3f} using {}\n".format(random_result.best_score_, random_result.best_params_)
        print(msg1)

        msg2 = "test_result: accuracy = {:.3f}, precision = {:.3f}, recall = {:.3f}, f1 = {:.3f}\n confusion_matrix = {}".format(accuracy, precision ,recall, f1, cf_matrix)
        print(msg2)

        # ベストパラメータでのスコア・パラメータ、及びテスト結果(recall, f1, confusion matrix)の記録
        with open (f'{res_path}/{file_stem}_rbf_random_result.txt',mode='a') as f:
            print(msg1, file=f)
            print(msg2, file=f)
        # msg = "{}\nAccuracy:{:.3f}\nPrecision:{:.3f}\nRecall:{:.3f}\nF1 Score:{:.3f}\n".format(file_name,results[0],results[1],results[2],results[3])

        means = random_result.cv_results_['mean_test_score']
        stds= random_result.cv_results_['std_test_score']
        params = random_result.cv_results_['params']

        df_result = pd.DataFrame(data=zip(means, stds, params), columns=['mean', 'std', 'params'])

        # スコアの降順に並び替え
        df_result = df_result.sort_values('std', ascending=True)
        df_result = df_result.sort_values('mean', ascending=False)

        df_result.to_csv(f'{res_path}/{file_stem}_rbf_random_result.csv')

        with open (f'{res_path}/{file_stem}_rbf_random_result.txt',mode='a') as f:
            # 各サーチにおけるスコア・標準偏差・パラメータを記録
            for index, row in df_result.iterrows():
                print("score: {:.3f} +/-{:.4f}, params: {}".format(row['mean'], row['std']*2, row['params']))

        # for mean, std, param in zip(means, stds, params):
        #     print("{:3f} ({:3f}) with: {}".format(mean, std, param))
        #     with open (f'{res_path}/{file_stem}_grid_result.txt',mode='a') as f:
        #         print("{:3f} ({:3f}) with: {}".format(mean, std, param), file=f)
        # if the numbers of features are less than CVEs with flag, recover the dropped CVEs with flag
        if (file_name == '02_retweet_count.csv'):
            df_flag_sorted = df_flag_temp


if __name__ == "__main__":
    main()

