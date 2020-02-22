import matplotlib.pyplot as plt
import random
import numpy as np
from skimage.io import imread
import matplotlib.patches as patches

from config import BASE_PATH
from utils import vis_utils

def vis_category(idx=None, name=None, cols=4, rows=4):
    idx_to_path, categories, cat_list = vis_utils('category')
    
    if idx is None:
        cur_lst = cat_list[categories.index(name)]
    else:
        cur_lst = cat_list[idx]
    random.shuffle(cur_lst)
    plt.figure(figsize=(8, 8))
    
    plt.suptitle('Category "%s"' % name, fontsize=22)
    for cur_col in range(cols):
        for cur_row in range(rows):
            idx = cur_row * cols + cur_col 
            plt.subplot(rows, cols, idx + 1)
            plt.axis('off')
            plt.imshow(imread(BASE_PATH + idx_to_path[cur_lst[idx]]))  
    plt.subplots_adjust(wspace=0.0, hspace=0.2)

def vis_img(path=None, idx=None):
    idx_to_path, bbox, categories, cat_target = vis_utils('img')
    
    plt.figure(figsize=(6, 8))
    if idx is not None:
        cur_path = idx_to_path[idx]
        cur_idx = idx
        category = categories[cat_target[cur_idx].argmax()]
    else:
        cur_path = path
        cur_idx = path_to_idx[path]
        category = categories[cat_target[cur_idx].argmax()]
    
    plt.imshow(imread(BASE_PATH + cur_path))

    p1, p2 = bbox[cur_idx][:2], bbox[cur_idx][2:]
    rect = patches.Rectangle((bbox[cur_idx][[0, 2]].min(), bbox[cur_idx][[1, 3]].min()), 
                             np.abs((p1-p2))[0], np.abs((p1-p2))[1], linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    plt.title('Category: %s\n' % category)
