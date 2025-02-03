import os
import jieba
import re
import pickle

FILE_PATH = './data_org'
RESULT_FILE_PATH = './data_proc/res.pkl'
file_list = os.listdir(FILE_PATH)
print(len(file_list) ,'Files Have found in' ,FILE_PATH)

text_all = list()
for file_name in file_list:
    with open('./data_org/' + file_name, 'r' ,encoding='utf-8') as fi:
        for line in fi:
            clean_stence = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "",line)
            if (len(clean_stence) != 0):
                x = jieba.lcut(clean_stence)
                x.append('<END>')
                x.insert(0,'<START>')
                text_all.append(x)
with open(RESULT_FILE_PATH, 'wb') as file_to_write:
    pickle.dump(text_all ,file_to_write)
        
