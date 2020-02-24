## *Deep Fashion* Dataset
[Deep Fashion Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html), especially "[Category and Attribute Prediction Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html)", was used for training.
It requires to follow download-instructions from *Deep Fashion* Dataset to use the dataset.

### Model
```sh
			-> Classification Head: (Dense->elu->Dense->softmax) -> category
InputImage -> Resnet50 
	                -> Regression Head: (Dense->relu->Dense->relu-> Dense) -> bbox(x1, y1, x2, y2)

```
### Loss function
```sh
Category: CrossEntropy
Bbox: MSE
```

### Train
```sh
python train.py
```

### Evaluate
```sh
python eval.py
```
### Example of training, evaluation and visualizaion on Google colab
```sh
main.ipynb
```

## Results

![Bbox](/imgs/2.png)


