from collections import defaultdict
import os
import re
import jieba
import codecs
#from snownlp import SnowNLP

def seg_word(sentence):  
    """使用jieba对文档分词"""    
    seg_list = jieba.cut(sentence)   
    seg_result = [element for element in seg_list]      
    #读取停用词文件 
    stopwords = set()    
    fr = codecs.open('stopwords.txt', 'r', 'utf-8')    
    for word in fr:        
        stopwords.add(word.strip()) 
    fr.close()    
    #去除停用词     
    word_list=list(filter(lambda x: x not in stopwords, seg_result))
    print(word_list)
    return word_list
      
#读取情感词典（需要根据词频统计进行扩充+网络词典的删减）
def create_sendic():   
    #读取国立台湾的大学情感词典
    sen_file1 = open('NTU_positive.txt', 'r+', encoding='utf-8')
    sen_file2 = open('NTU_negative.txt', 'r+', encoding='utf-8')
    #sen_file3 = open('sum_positive.txt', 'r+', encoding='utf-8')
    #sen_file4 = open('sum_negative.txt', 'r+', encoding='utf-8')
    sen_list1 = sen_file1.read().splitlines()
    sen_list2 = sen_file2.read().splitlines()
    #sen_list3 = sen_file3.read().splitlines()
    #sen_list4 = sen_file4.read().splitlines()
    print(len(sen_list1))
    print(len(sen_list2))
    sen_dict = defaultdict()
    #print(len(sen_list3))
    #print(len(sen_list4))
    '''#读取Boson网络情感词典
    sen_file = open('BosonNLP.txt', 'r+', encoding='utf-8') 
    sen_list = sen_file.read().splitlines()'''
    
    #整合字典
    for s in sen_list1:
        sen_dict[s] = 1
    for s in sen_list2:
        sen_dict[s] =-1
    #对每一行内容根据空格分隔，索引0是情感词，1是情感分值
    '''for s in sen_list:   
        if len(s.split(' ')) == 2:
            sen_dict[s.split(' ')[0]] = s.split(' ')[1]
    for key in sen_dict.keys():
        if float(sen_dict[key])>0:
            sen_dict[key]=1
        else:
            sen_dict[key]=-1'''
    return sen_dict

#读取否定词文件并创建列表
def create_notlist():
    not_word_file = open('notDic.txt', 'r+', encoding='utf-8')
    not_word_list = not_word_file.read().splitlines()
    return not_word_list

#获得每条评论的最终评分
def caculate_score(word_list,sen_dict,not_word_list):
    sen_score=dict()
    for i in range(0,len(word_list)): 
        if word_list[i] in sen_dict.keys():
            print(word_list[i])
            if word_list[i-1] in not_word_list or word_list[i-2] in not_word_list:
                sen_score[word_list[i]]=sen_dict[word_list[i]]*(-1)
                print("实际情绪指标",sen_dict[word_list[i]]*(-1))
            else:
                sen_score[word_list[i]]=sen_dict[word_list[i]]
                print("实际情绪指标",sen_dict[word_list[i]])
        else:
             sen_score[word_list[i]]=0
    sum_score=0        
    for score in sen_score.values():
        sum_score+=score
    if sum_score>0:
        final_score=1
    elif sum_score<0:
        final_score=-1
    else:
        final_score=0        
    print(final_score)
    return final_score

alist=['观望中','格力要完蛋，别买了']
socre_dic=dict()
for sentence in alist:    
    word_list=seg_word(sentence)
    sen_dict=create_sendic()
    not_word_list=create_notlist()
    final_score=caculate_score(word_list,sen_dict,not_word_list)
    socre_dic[sentence]=final_score
print(socre_dic)

alist=['格力要完蛋，别买了','今天吃了鸡蛋饼']
'''
for sentence in alist:  
    s = SnowNLP(sentence)
    print(s.sentiments) 
    print(s.keywords(3))
'''

