from PIL import Image
from skimage import transform
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import CATEGORIES, WIDTH, HEIGHT


def predict_on_img(model, file, upload_folder, filename):
    pil_img = Image.open(io.BytesIO(file)).convert("RGB")
    np_image = np.array(pil_img).astype('float32')
    np_image = transform.resize(np_image, (WIDTH, HEIGHT, 3))
    image = np.expand_dims(np_image, axis=0)

    pred_labels, pred_bboxs = model.predict(image)
    pred_bboxs = pred_bboxs[0] * WIDTH
    pred_category = CATEGORIES[pred_labels[0].argmax()]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(WIDTH, HEIGHT)
    plt.imshow(np_image.astype('uint8'))

    p1, p2 = pred_bboxs[:2], pred_bboxs[2:]
    rect = patches.Rectangle((pred_bboxs[[0, 2]].min(), pred_bboxs[[1, 3]].min()),
                             np.abs((p1 - p2))[0], np.abs((p1 - p2))[1], linewidth=1, edgecolor='r',
                             facecolor='none')
    plt.gca().add_patch(rect)

    plt.title(f'Category: {pred_category}')
    plt.axis('off')
    plt.savefig(f'{upload_folder}/{filename}')
