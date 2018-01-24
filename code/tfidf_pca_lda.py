# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import codecs
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.cross_validation import train_test_split 
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt 
from sklearn.decomposition import TruncatedSVD

def read_data(filename):
    f = open(filename, encoding = 'utf8')
    for line in f:
        print(line)
#    
#    print(f)
#    np1 = np.loadtxt(f, dtype = 'str', delimiter = ',')
#    np1 = np.loadtxt(filecp, dtype = 'str', delimiter = ',',skiprows=5)
    df1 = pd.read_csv(filename, header = None, encoding = "utf8")
    np1 = np.array(df1)
    data = np1[:, 0:-1]
    label = np1[:, -1]
    return data, label.astype('float32')
   
    
if __name__ == "__main__":
    data, label = read_data("../input/data_clean.csv")
    data_combine = np.hstack((data[:, 0], data[:, 1]))
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    label_up = np.hstack((label, label)).astype("float32")
    similarity = np.zeros((data.shape[0], 1))
    tfidf = transformer.fit_transform(vectorizer.fit_transform(data_combine))
    question = csr_matrix(tfidf[0:data.shape[0]])
    answer = csr_matrix(tfidf[data.shape[0]:])
    data_sparse = hstack([question, answer])
    plt.figure()
    data_sparse_one = PLSRegression(n_components = 100).fit(data_sparse.toarray(), label)
#    data_sparse_one = PCA(n_components = 100).fit(data_sparse.toarray())

    train_X,test_X, train_y, test_y = train_test_split(data_sparse_one, label, test_size = 0.2, random_state = 0)  
            
    for model in ['logistic', 'linear', 'GBDT', 'NN']:
        mse_train = []
        mse_test = []
        print(model)
        if model == 'NN':
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5, 
                         hidden_layer_sizes=(50, 10), random_state=1)
        n_estimator = 10          
#        clf = SVR(C=1.0, epsilon=0.2)
        if model == 'linear':
            clf = linear_model.LinearRegression()
        if model == 'logistic':
            clf = linear_model.LogisticRegression()
        if model == 'GBDT':
            clf = GradientBoostingClassifier(n_estimators=n_estimator)
        clf.fit(train_X, train_y) 
#            predict_train = clf.predict(train_X)
#            mse_train.append(mean_squared_error(predict_train, train_y))
#        print("error in train set", mean_squared_error(predict_train, train_y))
        predict_test = clf.predict(test_X)
        mse_test.append(mean_squared_error(predict_test, test_y))
    #        print("error in test set", mean_squared_error(predict_test, test_y))
#        plt.plot(range(len(mse_train)), mse_train, label = 'train')
        plt.plot(range(len(mse_test)), mse_test, label = model)
        plt.ylim(0.4,1.2)
    plt.legend()
    plt.show()
    