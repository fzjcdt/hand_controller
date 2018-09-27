# Hand Controller
---

## DEMO
![demo](./demo/demo.gif)

## TODO:
 - [x] Build an environment of tensorflow-gpu, opencv, etc.
 - [x] Build a data set and train a model.
 - [x] Load cpm and classification models to detect gestures and react accordingly.
 - [ ] Optimize multiple model loading.
 - [ ] The heat map is a grayscale map that does not require 3 channels, and using one channel to reduce the amount of calculation.
 - [ ] Add a layer of convolution and pooling layer to become AlexNet, the input size does not need to be compressed to 100*100.
 - [ ] Increase data sets and add other categories to improve generalization.
 - [ ] Pay more attention to the precision rate, ignore the recall rate, and remove the results with the closest predictions.

## REFERENCE
Improved on the basis of [Convolutional Pose Machinesm](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) and [HandGestureClassify](https://github.com/yyyerica/HandGestureClassify)




