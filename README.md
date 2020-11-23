### Installation
```sh
git clone https://github.com/adeshkin/deepfashion.git
cd deepfashion 
```
*Download weights from [gdrive](https://drive.google.com/drive/folders/1YzwgfZ5GkVpvSsm7uXwf6Hd3tCC1PZzP?usp=sharing)* to .
```sh
docker build -t deepfashion . && docker run -it -p 8008:8008 deepfashion 
```

## *Deep Fashion* Dataset
[Deep Fashion Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html), especially "[Category and Attribute Prediction Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html)", was used for training.
It requires to follow download-instructions from *Deep Fashion* Dataset to use the dataset.
## Paper
[DeepFashion: Powering Robust Clothes Recognition and Retrieval
with Rich Annotations](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf)
### Model structure
```sh
			 -> Classification Head: (Dense->elu->Dense->softmax) -> category
InputImage -> Resnet101* 
	                 -> Regression Head: (Dense->relu->Dense->relu->Dense) -> bbox(x1, y1, x2, y2)

*pretrained on ImageNet, just last 16 layers are trainable 
```

### *Model weights is available [here](https://drive.google.com/drive/folders/1YzwgfZ5GkVpvSsm7uXwf6Hd3tCC1PZzP?usp=sharing)*


### Loss function
```sh
Category: CrossEntropy
Bbox: MSE
```

### Train
```sh
python utils/train.py
```

## Results on test
### Image accuracy: 0.56
### Top-5 image accuracy: 0.87
### Bounding boxes error: 0.014
### Format of output:
#### *true category, predicted category*
#### *red - true bbox, blue - predicted bbox*
![Bbox1](/imgs/examples/10.png)
![Bbox2](/imgs/examples/example.png)


