from keras.models import Model
from keras.layers import Dense
from keras.applications.resnet import ResNet101
from keras.regularizers import l2


def create_model():
    model_resnet = ResNet101(weights='imagenet', include_top=False, pooling='avg')
    for layer in model_resnet.layers[:-16]:
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
