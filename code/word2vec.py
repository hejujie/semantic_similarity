# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import gensim

def read_data(filename):
    df1 = pd.read_csv(filename, header = None, encoding = "utf8")
    np1 = np.array(df1)
    data = np1[:, 0:-1]
    label = np1[:, -1]
    return data, label

#def 
    
if __name__ == "__main__":
    data, label = read_data("../input/data_clean.csv")
    data_combine = np.hstack((data[:, 0], data[:, 1]))
    for i in range(data_combine.shape[0]):
        data_combine[i] = ((data_combine[i].split(" ")))
#        break
#    a = [data_combine[5], data_combine[6]]
#    sentences = gensim.models.word2vec.LineSentence(data_combine[1:10])
    model = gensim.models.Word2Vec(data_combine, min_count=1, size=500, window=5, iter=500)
#    print(model['可'])
    print(model['可逆过程'])
    print(model.wv.most_similar(['伦敦','中国'],['北京']))
    print(model.wv.similarity('化学','物理'))
    model.save('../output/word2vec_500.txt')