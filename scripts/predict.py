from PIL import Image
from skimage import transform
import io


def predict_on_img(image_byte):
    img = Image.open(io.BytesIO(image.read())).convert("RGB")
    np_image = np.array(img).astype('float32')
    np_image = transform.resize(np_image, (224, 224, 3))
    image = np.expand_dims(np_image, axis=0)

    pred_labels, pred_bboxs = model.predict(image)

    pred_bboxs = pred_bboxs[0] * 224
    pred_label = CATEGORIES[pred_labels[0].argmax()]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(224, 224)
    plt.imshow(np_image.astype('uint8'))

    p1, p2 = pred_bboxs[:2], pred_bboxs[2:]
    rect = patches.Rectangle((pred_bboxs[[0, 2]].min(), pred_bboxs[[1, 3]].min()),
                             np.abs((p1 - p2))[0], np.abs((p1 - p2))[1], linewidth=1, edgecolor='r',
                             facecolor='none')
    plt.gca().add_patch(rect)
    plt.title('Category: %s\n' % pred_label)
    plt.axis('off')
    plt.savefig(app.config['UPLOAD_FOLDER'] + f'/{filename}')