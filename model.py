from keras.models import Model
from keras.layers import Dense
from keras.applications.resnet50 import ResNet50
from keras.regularizers import l2


def model():
    model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    for layer in model_resnet.layers[:-12]:
        # 6 - 12 - 18 have been tried. 12 is the best.
        layer.trainable = False

    x = model_resnet.output
    x = Dense(512, activation='elu', kernel_regularizer=l2(0.001))(x)
    y = Dense(46, activation='softmax', name='img')(x)

    x_bbox = model_resnet.output
    x_bbox = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x_bbox)
    x_bbox = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x_bbox)
    bbox = Dense(4, kernel_initializer='normal', name='bbox')(x_bbox)

    final_model = Model(inputs=model_resnet.input,
                        outputs=[y, bbox])
    return final_model