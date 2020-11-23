from PIL import Image
from skimage import transform
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import CATEGORIES, WIDTH, HEIGHT


def predict_on_img(model, file):
    pil_img = Image.open(io.BytesIO(file)).convert("RGB")
    np_image = np.array(pil_img).astype('float32')
    np_image = transform.resize(np_image, (WIDTH, HEIGHT, 3))
    image = np.expand_dims(np_image, axis=0)

    pred_labels, pred_bboxs = model.predict(image)
    pred_bbox = pred_bboxs[0] * WIDTH
    pred_category = CATEGORIES[pred_labels[0].argmax()]

    plt.figure(figsize=(4, 6))
    plt.axis('off')
    plt.imshow(np_image.astype('uint8'))

    p1, p2 = pred_bbox[:2], pred_bbox[2:]
    rect = patches.Rectangle((pred_bbox[[0, 2]].min(), pred_bbox[[1, 3]].min()),
                             np.abs((p1 - p2))[0], np.abs((p1 - p2))[1], linewidth=1, edgecolor='r',
                             facecolor='none')
    plt.gca().add_patch(rect)
    plt.title(f'Category: {pred_category}')
    f = io.BytesIO()
    plt.savefig(f)
    return f
