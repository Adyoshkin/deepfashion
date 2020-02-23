import numpy as np
from config import BASE_DIR

def get_path_id():
    path_to_idx = {}
    idx_to_path = {}
    bbox_file = open(BASE_DIR + 'anno/list_bbox.txt', 'r').readlines()
    bbox = [[int(v) if i > 0 else v for i, v in enumerate(list(filter(None, x[:-1].split(' '))))] for x in bbox_file[2:]]
    for i, cur_bbox in enumerate(bbox):
        path = cur_bbox[0]
        idx_to_path[i] = path
        path_to_idx[path] = i
    
    return path_to_idx, idx_to_path

def get_bbox():
    bbox_file = open(BASE_DIR + 'anno/list_bbox.txt', 'r').readlines()
    bbox = [[int(v) if i > 0 else v for i, v in enumerate(list(filter(None, x[:-1].split(' '))))] for x in bbox_file[2:]]
    bbox = np.array([ln[1:] for ln in bbox], dtype=np.int32)
    
    return bbox
    

def get_categories():
    cats_file = open(BASE_DIR + 'anno/list_category_cloth.txt', 'r').readlines()
    categories = []
    for ln in cats_file[2:]:
        cur = list(filter(None, ln[:-1].split(' ')))
        categories.append(cur[0])
    
    return categories


def vis_utils(flg):
    path_to_idx, idx_to_path = get_path_id()
    bbox = get_bbox()
    categories = get_categories()
    
    cats_img_file = open(BASE_DIR + 'anno/list_category_img.txt', 'r').readlines()
    cat_target = np.zeros((len(idx_to_path), len(categories)), dtype=np.uint8)
    cat_list = {}

    for ln in cats_img_file[2:]:
        cur = list(filter(None, ln[:-1].split(' ')))
        cur_cat = int(cur[1]) - 1

        cat_target[path_to_idx[cur[0]]][cur_cat] = 1
        if cur_cat in cat_list:
            cat_list[cur_cat].append(path_to_idx[cur[0]])
        else:
            cat_list[cur_cat] = [path_to_idx[cur[0]]]
            
    if flg == 'category':
        return idx_to_path, categories, cat_list
    else:
        return idx_to_path, bbox, categories, cat_target