# -*- coding: utf-8 -*-

import gensim
import pandas as pd
import numpy as np
#import tensorflow as tf

def read_data(filename):
    df1 = pd.read_csv(filename, header = None, encoding = "utf8")
    np1 = np.array(df1)
    data = np1[:, 0:-1]
    label = np1[:, -1]
    return data, label.astype("float32")
    
def data_embedding(data, model, word2vec_size, max_len):
    embedding = np.zeros((max_len, word2vec_size))
    for index, word in enumerate(data):
        try:
            embedding[index] = model[word]
        except:
#            print("except")
            pass
    embedding = embedding.reshape(1, -1)
    return embedding        
        
        


if __name__ == "__main__":
    data, label = read_data("../input/data_clean.csv")
    data_combine = np.hstack((data[:, 0], data[:, 1]))
    for i in range(data_combine.shape[0]):
        data_combine[i] = ((data_combine[i].split(" ")))
        
    word2vec_size = 64
    max_len = 20
#    model = gensim.models.KeyedVectors.load_word2vec_format("../input/news_12g_baidubaike_20g_novel_90g_embedding_64.bin", binary = True)
#    model.train(data_combine)
    #    model = gensim.models.Word2Vec.load("../output/word2vec_500.txt")
    embedding_output = np.zeros((data_combine.shape[0], word2vec_size*max_len))
    for i in range(data_combine.shape[0]):
        embedding_output[i] = data_embedding(data_combine[i], model, word2vec_size, max_len)
    embedding_output = np.hstack((embedding_output[0:data.shape[0]], embedding_output[data.shape[0]:]))
    df1 = pd.DataFrame(embedding_output.astype('float32'))
    df2 = pd.DataFrame(label.astype('float32'))
    output = pd.concat([df1, df2], axis = 1)
    print("begin to output")
    output.to_csv("../output/embedding_all_60.csv", index = None, header = None, float_format = "%.6f")
#    del output, df1, df2, embedding_output
