import matplotlib.pyplot as plt
import random
from keras.models import load_model
import numpy as np
from skimage.io import imread, imsave
import matplotlib.patches as patches
from keras.preprocessing.image import ImageDataGenerator   
import os
from get_bbox import get_dict_bboxes
from dataloader import DirectoryIteratorWithBoundingBoxes
from config import BASE_DIR, SAVE_DIR, CATEGORIES, TEST_DIR, TARGET_SIZE, MODEL_PATH
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
            plt.imshow(imread(BASE_DIR + idx_to_path[cur_lst[idx]]))  
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
    
    plt.imshow(imread(BASE_DIR + cur_path))

    p1, p2 = bbox[cur_idx][:2], bbox[cur_idx][2:]
    rect = patches.Rectangle((bbox[cur_idx][[0, 2]].min(), bbox[cur_idx][[1, 3]].min()), 
                             np.abs((p1-p2))[0], np.abs((p1-p2))[1], linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    plt.title('Category: %s\n' % category)

def vis_pred(cols=4, rows=4):
    model = load_model(MODEL_PATH)
    
    dict_test = get_dict_bboxes(mode='test')
    test_datagen = ImageDataGenerator()
    test_iterator = DirectoryIteratorWithBoundingBoxes(TEST_DIR, test_datagen, 
                                                       bounding_boxes=dict_test, target_size=TARGET_SIZE)

    imgs, (labels, bboxs) =  test_iterator.next()
    pred_labels, pred_bboxs = model.predict_on_batch(imgs)
    bboxs = bboxs * 200
    pred_bboxs = pred_bboxs * 200
    
    os.mkdir(SAVE_DIR)
    for i, img in enumerate(imgs):
        imsave(f'{SAVE_DIR}/{i}.png', img.astype(np.uint8))
        
    plt.figure(figsize=(16, 16))    
    for cur_col in range(cols):
        for cur_row in range(rows):
            idx = cur_row * cols + cur_col
            ind = idx
            bbox = bboxs[ind]
            pred_bbox = pred_bboxs[ind]
            label = CATEGORIES[labels[ind].argmax()]
            pred_label = CATEGORIES[pred_labels[ind].argmax()]
            plt.subplot(rows, cols, idx + 1)
            plt.axis('off')
            plt.title(f"true: {label}, pred: {pred_label}")
            plt.imshow(imread(f'{SAVE_DIR}/{ind}.png'))
            p1, p2 = bbox[:2], bbox[2:]
            rect = patches.Rectangle((bbox[[0, 2]].min(), bbox[[1, 3]].min()), 
                                      np.abs((p1-p2))[0], np.abs((p1-p2))[1], linewidth=1, edgecolor='r', facecolor='none')

            pred_p1, pred_p2 = pred_bbox[:2], pred_bbox[2:]
            pred_rect = patches.Rectangle((pred_bbox[[0, 2]].min(), pred_bbox[[1, 3]].min()), 
                                          np.abs((pred_p1-pred_p2))[0], np.abs((pred_p1-pred_p2))[1], 
                                          linewidth=1,edgecolor='b', facecolor='none')
            plt.gca().add_patch(rect)
            plt.gca().add_patch(pred_rect)
            plt.subplots_adjust(wspace=0.0, hspace=0.2)
