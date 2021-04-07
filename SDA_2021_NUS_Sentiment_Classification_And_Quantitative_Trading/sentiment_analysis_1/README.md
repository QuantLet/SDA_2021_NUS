[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **sentiment_analysis_1** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet: 'sentiment_analysis_1'

Published in: 'SDA_2021_NUS'

Description: 'Project_Group8 file'

Keywords: 'NUS, FE5225, sentiment analysis, stopwords, first step'

Author: 'Wang Yuzhou'

```

### PYTHON Code
```python

import os
import jieba
from collections import OrderedDict
import numpy as np
import pandas as pd
import csv


with open('stopwords.txt', 'r+', encoding='utf-8') as f:
    stop_words = set(f.read().splitlines())

with open('notDic.txt', 'r+', encoding='utf-8') as f:
    not_words = set(f.read().splitlines())

with open('NTU_positive.txt', 'r+', encoding='utf-8') as f:
    NTU_positive = set(f.read().splitlines())
with open('NTU_negative.txt', 'r+', encoding='utf-8') as f:
    NTU_negative = set(f.read().splitlines())

# read manual dictionary
manual_df = pd.read_csv('manual_dict.csv')
manual_pos = set(manual_df[manual_df.value == 1].word)
manual_neg = set(manual_df[manual_df.value == -1].word)

pos_set = NTU_positive
neg_set = NTU_negative

pos_set.update(manual_pos)
neg_set.update(manual_neg)

sen_dict = dict()
for word in pos_set:
    sen_dict[word] = 1
for word in neg_set:
    sen_dict[word] =-1

def seg_word(sentence):  
    """使用jieba对文档分词"""    
    seg_list = list(jieba.cut(sentence))         
    #去除停用词     
    word_list=list(filter(lambda x: x not in stop_words, seg_list))
    return word_list

def calculate_score_by_list(word_list):
    sen_index_list = list(map(lambda x: sen_dict.get(x,0),word_list))
    not_word_index_list = list(map(lambda x: x in not_words,word_list))
    if len(not_word_index_list) >= 2:
        roll_1 = np.roll(not_word_index_list,1)
        roll_2 = np.roll(not_word_index_list,2)
        roll_1[0] = 0
        roll_2[0] = 0
        roll_2[1] = 0
        not_word_index_list = -1 * np.logical_xor(roll_1,roll_2)
        not_word_index_list[not_word_index_list == 0] = 1
    elif len(not_word_index_list) >= 1:
        roll_1 = np.roll(not_word_index_list,1)
        roll_1[0] = 0
        not_word_index_list = -1 * roll_1
        not_word_index_list[not_word_index_list == 0] = 1
    else:
        return 0
    score = np.sum(sen_index_list * not_word_index_list)
    final_score = 1 if score > 0 else -1 if score < 0 else 0
    return final_score

def score(sentence):
    word_list = seg_word(sentence)
    score_ = calculate_score_by_list(word_list)
    return score_

# read data
df = pd.read_csv('merge_data.csv')

'''
# This part is to build dictionary manually, whose name is manual_dict.csv
# word count
word_count_dict = dict()
for num, sentence in enumerate(df['text_a']):
    if type(sentence) is str:
        word_list = seg_word(sentence)
        for word in word_list:
            word_count_dict[word] = word_count_dict.get(word,0) + 1
    if num % 10000 == 0:
        print(num)

# enter value by the order of the count of key
order_word_cout_dict = OrderedDict(sorted(word_count_dict.items(),key=lambda item:item[1],reverse=True))
dict_df = list()
for key in order_word_cout_dict.keys():
    quit = False
    if key not in stop_words and key not in not_words and key not in NTU_positive and key not in NTU_negative:
        while True:
            print(f'{key} with frequency {order_word_cout_dict[key]} times')
            print(f'Whats the score of {key}? 1 for pos, -1 for neg, 0 for neither, q for stop and quit ! ')
            score = input()
            if score == '1' or score == '-1' or score == '0':
                dict_df.append([key,order_word_cout_dict[key], int(score)])
                break
            elif score == 'q':
                quit = True
                break
    if quit:
        break

# save manual dictionary
dict_df = pd.DataFrame(dict_df, columns=['word','count','value'])
dict_df.to_csv('manual_dict.csv')
'''

# calculate score_list
score_list = list(map(lambda content:0 if type(content) is float else score(content), df['text_a']))


# scoring
single_stock_manual_score = pd.DataFrame(columns=['stock_id','news_id','score'])



```

automatically created on 2021-04-01