from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard

from get_bbox import get_dict_bboxes
from dataloader import DirectoryIteratorWithBoundingBoxes
from model import model

def custom_generator(iterator):
    while True:
        batch_x, batch_y = iterator.next()
        yield (batch_x, batch_y)

        
def train():
    final_model = model()
    opt = SGD(lr=0.0001, momentum=0.9, nesterov=True)

    final_model.compile(optimizer=opt,
                        loss={'img': 'categorical_crossentropy',
                              'bbox': 'mean_squared_error'},
                        metrics={'img': ['accuracy', 'top_k_categorical_accuracy'], # default: top-5
                                 'bbox': ['mse']})
    
    train_datagen = ImageDataGenerator(rotation_range=30.,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator()

    dict_train, dict_val, dict_test = get_dict_bboxes()
    train_iterator = DirectoryIteratorWithBoundingBoxes("./data/img/train", train_datagen, 
                                                        bounding_boxes=dict_train, target_size=(200, 200))
    test_iterator = DirectoryIteratorWithBoundingBoxes("./data/img/val", test_datagen, 
                                                       bounding_boxes=dict_val, target_size=(200, 200))
    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                   patience=12,
                                   factor=0.5,
                                   verbose=1)
    tensorboard = TensorBoard(log_dir="/content/drive/My Drive/deepfashion/logs")
    early_stopper = EarlyStopping(monitor='val_loss',
                                  patience=30,
                                  verbose=1)
    checkpoint = ModelCheckpoint("/content/drive/My Drive/deepfashion/models/model.h5")

    final_model.fit_generator(custom_generator(train_iterator),
                              steps_per_epoch=2000,  
                              epochs=200, 
                              validation_data=custom_generator(test_iterator),
                              validation_steps=200,
                              verbose=2,
                              shuffle=True,
                              callbacks=[lr_reducer, checkpoint, early_stopper, tensorboard])
if __name__ == '__main__':
    train()