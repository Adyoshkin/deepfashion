from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from dataset.dataloader import DirectoryIteratorWithBoundingBoxes
from utils.utils import get_dict_bboxes
from config import TEST_DIR, MODEL_PATH, TARGET_SIZE, STEPS


def custom_generator(iterator):
    while True:
        batch_x, batch_y = iterator.next()
        yield (batch_x, batch_y)


def evaluate():
    dict_test = get_dict_bboxes(mode='test')
    test_datagen = ImageDataGenerator()
    test_iterator = DirectoryIteratorWithBoundingBoxes(TEST_DIR,
                                                       test_datagen,
                                                       shuffle=False,
                                                       bounding_boxes=dict_test,
                                                       target_size=TARGET_SIZE)
    
    model = load_model(MODEL_PATH)
    scores = model.evaluate_generator(custom_generator(test_iterator),
                                      steps=STEPS)

    print('Multi target loss: ' + str(scores[0]))
    print('Image loss: ' + str(scores[1]))
    print('Bounding boxes loss: ' + str(scores[2]))
    print('Image accuracy: ' + str(scores[3]))
    print('Top-5 image accuracy: ' + str(scores[4]))
    print('Bounding boxes error: ' + str(scores[5]))


if __name__ == '__main__':
    evaluate()
