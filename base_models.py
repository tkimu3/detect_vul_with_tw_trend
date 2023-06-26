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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# train_test_split is used to split the data into train and test set of given data
# cross_val_score is used to find the score on given model and KFlod
# KFold is used for defining the no.of folds for Cross Validation
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# import pipeline to make machine learning pipeline to overcome data leakage problem
from sklearn.pipeline import Pipeline

# import StandardScaler to Column Standardize the data
# many algorithm assumes data to be Standardized
from sklearn.preprocessing import StandardScaler

# import warnings filter
from warnings import simplefilter

from pathlib import Path, PurePath

def main():

    # ignore all warnings
    simplefilter(action='ignore', category=Warning)

    ## Read the csv files
    # get current relative path
    cur_dir = Path('.')

    # get current absolute path
    # cur_dir = Path.cwd()

    # feeatures directory
    data_dir = 'features'

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

    for i, filepath in enumerate(filePaths):
        # extract filename from filepath
        file_name = filepath.name
        print('\n'+ file_name)

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

        ## Create a Validation dataset
        # Define parameters for each model
        validation_size = 0.30
        seed = 1234
        num_splits = 10
        scorings = ['accuracy','precision','recall','f1']

        # split data into train and test data
        train_data = df[i]
        target_data = df_flag_sorted['flag_H']

        # .values returns numpy.ndarray
        # using array instead of dataframe to train model because
        # array is faster to compute instead of dataframe
        X = train_data.values
        y = target_data.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = validation_size, random_state = seed)


        ## Evaluate Algorithms: Baseline
        # Spot-check Algorithms
        models = []

        # In LogisticRegression set: solver='lbfgs',multi_class ='auto', max_iter=10000 to overcome warning
        models.append(('LR',LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=10000)))
        models.append(('LDA',LinearDiscriminantAnalysis()))
        models.append(('KNN',KNeighborsClassifier()))
        models.append(('CART',DecisionTreeClassifier()))
        models.append(('NB',GaussianNB()))
        models.append(('SVM',SVC(gamma='scale')))
        models.append(('RF', RandomForestClassifier()))

        for scoring in scorings:
            # evaluate each model in turn
            results = []
            names = []
            print('\n'+ scoring)

            with open (f'{file_name}_{scoring}_algorithm_comparison.txt',mode='a') as f:
                print('\n'+ scoring, file=f)

            # evaluate each model in turn
            for name, model in models:

                # StratifiedKFoldの設定
                # initializing kfold by n_splits=10(no.of K)
                skf = StratifiedKFold(n_splits = num_splits, shuffle = True, random_state = seed)

                cv_results = cross_val_score(model, X_train, y_train, cv=skf, scoring=scoring )
                results.append(cv_results)
                names.append(name)

                msg = "{}:{:.3f}({:.3f})".format(name, cv_results.mean(), cv_results.std() )
                print(msg)

                with open (f'{file_name}_{scoring}_algorithm_comparison.txt',mode='a') as f:
                    print(msg, file=f)
                    print(str(cv_results.tolist()), file=f)
                    print('\n', file=f)
                    # f.write(msg)
                    # f.writelines(str(cv_results.tolist()))

            # Compare Algorithms
            fig = plt.figure()
            fig.suptitle(f'{file_name} {scoring} Algorithm Comparison')
            ax = fig.add_subplot(111)
            plt.boxplot(results)
            ax.set_xticklabels(names)
            plt.show()
            fig.savefig(f"{file_name}_{scoring}_Algorithm_Comparison.png")


        # if the numbers of features are less than CVEs with flag, recover the dropped CVEs with flag
        if (file_name == '02_retweet_count.csv'):
            df_flag_sorted = df_flag_temp

    # ## Evaluate Algorithms: Standardize Data
    # # Define pipeline for each base model

    # pipelines = []
    # pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000))])))
    # pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
    # pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
    # pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
    # pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])))
    # pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC(gamma='scale'))])))
    # pipelines.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF',RandomForestClassifier())])))

    # # evaluate each model in turn
    # scaled_results = []
    # scaled_names = []

    # # evaluate each model in turn
    # for name, model in pipelines:

    #     # StratifiedKFoldの設定
    #     # initializing kfold by n_splits=10(no.of K)
    #     skf = StratifiedKFold(n_splits = num_splits, shuffle = True, random_state = seed)

    #     cv_results = cross_val_score(model, X_train, y_train, cv=skf, scoring=scoring )
    #     scaled_results.append(cv_results)
    #     scaled_names.append(name)

    #     msg = "{}:{:.3f}({:.3f})".format(name, cv_results.mean(), cv_results.std() )
    #     print(msg)
    #     with open ('results_scaled_algorithm_comparison.txt',mode='a') as f:
    #         print(msg, file=f)
    #         print(str(cv_results.tolist()), file=f)
    #         print('\n', file=f)
    #         # f.write(msg)
    #         # f.writelines(str(cv_results.tolist()))


    # # Compare Algorithms
    # fig = plt.figure()
    # fig.suptitle('Algorithm Comparison (Scaled)')
    # ax = fig.add_subplot(111)
    # plt.boxplot(scaled_results)
    # ax.set_xticklabels(scaled_names)
    # plt.show()
    # fig.savefig("Algorithm_Comparison_Scaled.png")


if __name__ == "__main__":
    main()

