import body_recognition
import os
import json
DATA_PATH = '.\\original_data'
OUTPUT_PATH = '.\\train_data'
# 递归方法遍历所有待处理的视频文件
def list_child_dir(file_path):
    try:
        child_names = os.listdir(file_path)
    except:
        # 如果没有子文件，则被异常捕获，并返回
        bone_detector = body_recognition.Body_recognition(file_path)
        log = bone_detector.detect()
        data_list = list()
        for i in log:
            data_list.append(i)
            #print(i)
        json.dumps(data_list)
        with open(file_path.replace(DATA_PATH ,OUTPUT_PATH), 'w') as f:
            f.write(json.dumps(data_list))
        return None
    for name in child_names:
        list_child_dir(file_path + '\\' + name)

list_child_dir(DATA_PATH)
