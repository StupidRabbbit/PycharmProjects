#  -*- coding: UTF-8 -*-
import jieba
import os
import sys
import codecs
import uniout
import json

def TextProcessing(filepath):
    folder_list=os.listdir(filepath)
    data_list=[]
    class_list=[]

    for foler in folder_list:
        new_folder_path=os.path.join(filepath,foler)
        files=os.listdir(new_folder_path)
        j=1
        for file in files:
            if j>100:
                break
            with open(os.path.join(new_folder_path,file),'r') as f:
              lines=f.read()
            word_cut=jieba.cut(lines,cut_all=False)
            # print ','.join(word_cut)
            word_list=list(word_cut)
            # word_list=json.dumps(word_list,encoding='UTF-8',ensure_ascii=False)

            data_list.append(word_list)
            class_list.append(foler)
            j+=1
    print type(data_list)
    # data_list = json.dumps(data_list, encoding='UTF-8', ensure_ascii=False)

    # print data_list
    for data in data_list:
        print data
    # print class_list

if __name__ == '__main__':

    filepath='D:\CUIblog\Machine-Learning-master\Machine-Learning-master\Naive Bayes\SogouC\Sample'
    classpath='D:\CUIblog\Machine-Learning-master\Machine-Learning-master\Naive Bayes\SogouC\ClassList.txt'
    TextProcessing(filepath)