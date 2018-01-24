# -*- coding: utf-8 -*-

from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import jieba

def cut_sentence(content):  
    stop_word = get_stopword("../input/stop_words.txt")
    for i in range(content.shape[0]):
        output = ""
        sentence = jieba.cut(content[i])
        word_num = 0
        for word in sentence:
            word_num += 1
            if word not in stop_word:
                output = output + word + " "
            if word_num > 50:
                break
        content[i] = output
    return content
 
def get_stopword(filename):
    with open(filename, 'rb') as fopen:
        stop_content = fopen.read()
        stopword_list = stop_content.splitlines()
    down_1 = "_"
    down_n = ""
    for i in range(100):
        down_n += down_1
        stopword_list.append(down_n)
    return stopword_list
    
            
def read_data(filename):
    df1 = pd.read_csv(filename, header = None, sep = '\t')
    df1 = df1.dropna()
    number = np.array(df1[2].apply(lambda x: x.isdigit()))
    index = np.where(number == True)[0]
    data = np.array(df1[[0, 1]])[index]
    label = np.array(df1[2])[index].astype('float32')  
    print(label.shape, data.shape)
    return data, label

    
if __name__ == "__main__":
    data, label = read_data("../input/raw_train_data.txt")
    data[:, 0] = cut_sentence(data[:, 0])
    data[:, 1] = cut_sentence(data[:, 1])
    
    print(data.shape)
    df3 = pd.DataFrame(data[:, 0])
    df5 = pd.DataFrame(data[:, 1])
    df4 = pd.DataFrame(label)
    output = pd.concat([df3, df5, df4], axis = 1)
    print(len(output))
    output.to_csv("../input/data_clean.csv", index = None, header = False)
  

    
    

