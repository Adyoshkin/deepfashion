import shutil
import os
import re
from config import IMG_DIR


def split_data():
    splitter = re.compile("\s+")
    
    with open('../data/anno/list_eval_partition.txt', 'r') as eval_partition_file:
        list_eval_partition = [line.rstrip('\n') for line in eval_partition_file][2:]
        list_eval_partition = [splitter.split(line) for line in list_eval_partition]
        list_all = [(v[0][4:], v[0].split('/')[1].split('_')[-1], v[1]) for v in list_eval_partition]
        
    for element in list_all:
        if not os.path.exists(os.path.join(IMG_DIR, element[2])):
            os.mkdir(os.path.join(IMG_DIR, element[2]))
        if not os.path.exists(os.path.join(os.path.join(IMG_DIR, element[2]), element[1])):
            os.mkdir(os.path.join(os.path.join(IMG_DIR, element[2]), element[1]))
        if not os.path.exists(os.path.join(os.path.join(os.path.join(os.path.join(IMG_DIR, element[2]), element[1])),
                              element[0].split('/')[0])):
            os.mkdir(os.path.join(os.path.join(os.path.join(os.path.join(IMG_DIR, element[2]), element[1])),
                                  element[0].split('/')[0]))
        shutil.move(os.path.join(IMG_DIR, element[0]),
                    os.path.join(os.path.join(os.path.join(IMG_DIR, element[2]), element[1]), element[0]))


if __name__ == '__main__':
    split_data()
