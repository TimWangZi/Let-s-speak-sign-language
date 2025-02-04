import body_recognition
import os
import sys
import json
import time
DATA_PATH = '.\\original_data'
OUTPUT_PATH = '.\\train_data'
# 递归方法遍历所有待处理的视频文件
def list_child_dir(file_path):
    try:
        child_names = os.listdir(file_path)
    except:
        # 如果没有子文件，则被异常捕获，并返回

        if os.path.exists(file_path.replace(DATA_PATH ,OUTPUT_PATH).replace('.mp4' ,'.json')):
            print('[skip]:' ,file_path.replace(DATA_PATH ,OUTPUT_PATH).replace('.mp4' ,'.json'))
            return None
        
        bone_detector = body_recognition.Body_recognition(file_path)
        log = bone_detector.detect()
        data_list = list()
        # 遍历帧
        for i in log:
            data_list.append(i)
            #print(i)
        # 如果不存在文件则创建
        if not os.path.exists(file_path[0:file_path.rfind('\\')].replace(DATA_PATH ,OUTPUT_PATH)):
            os.makedirs(file_path[0:file_path.rfind('\\')].replace(DATA_PATH ,OUTPUT_PATH))
        with open(file_path.replace(DATA_PATH ,OUTPUT_PATH).replace('.mp4' ,'.json'), 'w') as f:
            f.write(json.dumps(data_list))
        print('[done]:', file_path)
        return None
    for name in child_names:
        list_child_dir(file_path + '\\' + name)
T_0 = time.time()
list_child_dir(DATA_PATH)
T_1 = time.time()
print('Finish the data preprocessing in :',(T_1 - T_0) / (60 * 1000),'Minutes')
