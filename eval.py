from keras.preprocessing.image import ImageDataGenerator
from dataloader import DirectoryIteratorWithBoundingBoxes

def evaluate():
    test_datagen = ImageDataGenerator()

    test_iterator = DirectoryIteratorWithBoundingBoxes("./img/test", test_datagen,
                                                       bounding_boxes=dict_test, target_size=(200, 200))
    scores = model.evaluate_generator(custom_generator(test_iterator), steps=2000)

    print('Multi target loss: ' + str(scores[0]))
    print('Image loss: ' + str(scores[1]))
    print('Bounding boxes loss: ' + str(scores[2]))
    print('Image accuracy: ' + str(scores[3]))
    print('Top-5 image accuracy: ' + str(scores[4]))
    print('Bounding boxes error: ' + str(scores[5]))