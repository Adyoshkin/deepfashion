from keras.preprocessing.image import ImageDataGenerator
import os
from keras.models import load_model
from skimage.io import imsave
from get_bbox import get_dict_bboxes
from dataloader import DirectoryIteratorWithBoundingBoxes
from config import TEST_DIR, TARGET_SIZE, MODEL_PATH, SAVE_DIR, CATEGORIES

def vis_pred():
    model = load_model(MODEL_PATH)

    imgs, (labels, bboxs) =  test_iterator.next()
    pred_labels, pred_bboxs = model.predict_on_batch(imgs)
    bboxs = bboxs * 200
    pred_bboxs = pred_bboxs * 200
    dict_test = get_dict_bboxes(mode='test')
    test_datagen = ImageDataGenerator()
    test_iterator = DirectoryIteratorWithBoundingBoxes(TEST_DIR, test_datagen, 
                                                       bounding_boxes=dict_test, target_size=TARGET_SIZE)
    
    os.mkdir(SAVE_DIR)
    for i, img in enumerate(imgs):
        imsave(f'{SAVE_DIR}/{i}.png', img.astype(np.uint8))
        