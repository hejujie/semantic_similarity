# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
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
import matplotlib.pyplot as plt 

def read_data(filename):
#    f = open(filename, encoding = 'gbk')
#    f.readline()
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
    for model in ['logistic', 'linear', 'GBDT', 'NN']:
        mse_train = []
        mse_test = []
        print(model)
        for i in range(1, 100):
            data_sparse_one = SelectKBest(chi2, k = i*100).fit_transform(data_sparse, label)
    #        print("precess finished")
            
            train_X,test_X, train_y, test_y = train_test_split(data_sparse_one, label, test_size = 0.2, random_state = 0)  
            
        
        #    similarity_all = cosine_similarity(question, answer)
        #    for i in range(similarity_all.shape[0]):
        #        similarity[i] = similarity_all[i, i]
        
        #    for i in range(data.shape[0]):
        #        try:
        #            tfidf = transformer.fit_transform(vectorizer.fit_transform(data[i, :]))
        #            similarity[i] = (cosine_similarity(tfidf)[0, 1])
        #        except:
        #            similarity[i] = 0
        #            label[i] = 0
        #            print(data[i])
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
    