from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from dataloader import DirectoryIteratorWithBoundingBoxes
from get_bbox import get_dict_bboxes

def custom_generator(iterator):
    while True:
        batch_x, batch_y = iterator.next()
        yield (batch_x, batch_y)
        
def predict():
    model = load_model("/content/drive/My Drive/deepfashion/models/model.h5")
    
    dict_train, dict_val, dict_test = get_dict_bboxes()
    test_datagen = ImageDataGenerator()
    test_iterator = DirectoryIteratorWithBoundingBoxes("./img/test", test_datagen, bounding_boxes=dict_test, 
                                                       target_size=(200, 200))
    
    preds = model.predict_generator(custom_generator(test_iterator))
    
    return preds