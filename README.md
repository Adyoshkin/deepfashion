## *Deep Fashion* Dataset
[Deep Fashion Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html), especially "[Category and Attribute Prediction Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html)", was used for training.
It requires to follow download-instructions from *Deep Fashion* Dataset to use the dataset.

### Train
```sh
python train.py
```

### Evaluate
```sh
python eval.py
```
### MODEL
```sh
						                            ->	Classification Head (Categories)
InputImage	->	VGG16 + Layers	--
						                            ->	Regression Head	(Confidnence in the Classification head prediction)

```

## Examples

![Bbox](/imgs/2.png)


