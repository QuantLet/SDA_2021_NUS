[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **sentiment_analysis_2** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet: 'sentiment_analysis_2'

Published in: 'SDA_2021_NUS'

Description: 'Project_Group8 file'

Keywords: 'NUS, FE5225, sentiment analysis, deep learning, second step'

Author: 'Wang Yuzhou'

```

### PYTHON Code
```python

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 22:38:42 2019

@author: lenovo
"""
import pandas as pd
import numpy as np
import gensim
import os
import jieba
import codecs
#import pydot
from gensim.models import word2vec
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from keras.utils import to_categorical,print_summary
from keras.utils.vis_utils import plot_model
from itertools import chain 

#读取并预处理数据
def seg_word(sentences):
    stopwords = set()    
    fr = codecs.open('stopwords.txt', 'r', 'utf-8')    
    for word in fr:        
        stopwords.add(word.strip()) 
    fr.close()    
    seg_list=[]
    for sentence in sentences:
        per_gen = jieba.cut(sentence)
        per_seg=[element for element in per_gen]
        #去除停用词
        filter_seg=list(filter(lambda x: x not in stopwords, per_seg))
        seg_list.append(filter_seg)
    #print(seg_list)
   # seg_result = list(chain(*seg_list))
    #print(seg_result)
    #读取停用词文件 

    #去除停用词     
    #word_list=list(filter(lambda x: x not in stopwords, seg_list))
    print(seg_list[0:20])
    return seg_list


#tst=pd.Series(["中石化给大家发福利了","这股完了，别再唱多了"])
#seg_word(tst)


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname),encoding='utf-8'):
                yield seg_word(str(line.split()))


def word2vec_model(sentences):
    #sentences = MySentences('datasets') # a memory-friendly iterator
    word_list=seg_word(sentences)
    # 嵌入的维度
    embedding_vector_size =256
    model = gensim.models.Word2Vec(word_list,min_count=5,size=embedding_vector_size)
    #model['生活']
    #model.most_similar(['生活'])
    # 取得所有单词
    vocab_list = list(model.wv.vocab.keys())
   # print(vocab_list)
    # 每个词语对应的索引
   # word_index = Dictionary()
    #word_index.doc2bow(model.wv.vocab.keys(),allow_update=True)
   # print(word_index)
   # w2indx = {v: k+1 for k, v in word_index.items()}#所有频数超过10的词语的索引
   # print(w2indx)
   # w2vec = {word: model[word] for word in w2indx.keys()}
    word_index = {word: index for index, word in enumerate(vocab_list)}
    model.save("word2vec.model")
    return model,word_index

# 序列化
def get_index(sentence):
    global word_index
    sequence = []
    for word in sentence:
        try:
            sequence.append(word_index[word])
        except KeyError:
            pass
    return sequence

def get_split_set(data,sentences):
    X_data = list(map(get_index, sentences))    
    # 截长补短
    maxlen =256
    X_pad = pad_sequences(X_data, maxlen=maxlen)
    # 取得标签
    Y = to_categorical(np.array(data['label']),num_classes = 3)
    # 划分数据集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_pad,
        Y,
        test_size=0.2,
        random_state=42)
    return X_train, X_test, Y_train, Y_test


#构建分类模型
# 让 Keras 的 Embedding 层使用训练好的Word2Vec权重
def LSTM_model(model,X_train, X_test, Y_train, Y_test):
    maxlen = 256
    embedding_matrix =model.wv.vectors
    model = Sequential()
    model.add(Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        input_length=maxlen))
       # weights=[embedding_matrix],
       # trainable=True))
    model.add(Bidirectional(LSTM(128, recurrent_dropout=0.1,return_sequences = True)))
    model.add(Bidirectional(LSTM(128, recurrent_dropout=0.1)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    model.compile(
        loss="categorical_crossentropy",
        optimizer='adam',
        metrics=['accuracy'])
    model_fit = model.fit(
        x=X_train,
        y=Y_train,
        validation_data=(X_test, Y_test),
        batch_size=20,
        epochs=8) 
    plot_model(model, to_file='lstm_model.png')
    return model_fit


if __name__ == "__main__":
    data=pd.read_csv('merge_data.csv',encoding='utf-8').dropna(axis=0, how='any')
    sentences=data['text_a'].astype(str)
    model,word_index=word2vec_model(sentences)
    X_train, X_test, Y_train, Y_test=get_split_set(data,sentences)
    model_fit=LSTM_model(model,X_train, X_test, Y_train, Y_test)

data=pd.concat([data['label'].astype(int),data['text_a']],axis=1)
train_data=data[0:2386]
dev_data=data[2386:3181]
test_data=data[3181:3977]
train_data.to_csv("train_data",sep='\t',index=False)
dev_data.to_csv("dev_data",sep='\t',index=False)
test_data.to_csv("test_data",sep='\t',index=False)
print(type(data['label'][0]))
'''
#加载模型并形成初始权重矩阵
#加载训练好的模型

embedding_model = word2vec.Word2Vec.load("word2vec.model")
#模型中有的单词向量就用该向量表示，没有的则随机初始化一个向量 
embedding_weights = np.array(embedding_model[word] if word in embedding_model else                              
                             np.random.uniform(-0.25, 0.25, embedding_model.vector_size)                
                             for word in embedding_model.wv.vocab.items())

cal_len = pd.DataFrame()
cal_len['post_length'] = list(map(len, sentences))
print("中位数：", cal_len['post_length'].median())
print("均值数：", cal_len['post_length'].mean())
#del cal_len'''





```

automatically created on 2021-04-01