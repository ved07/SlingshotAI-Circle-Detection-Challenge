# Circle Detection Challenge set by SlingshotAI
This repository contains the models used for the SlingshotAI circle detection challenge. 

My final results had an Intersection over Union accuracy of 0.9005 (90.05%) over 100 test examples, with a model consisting of 11,803,003 parameters, trained for 99 epochs, over 80,000 examples.

I implented 3 models, SimpleNet, NotSoSimpleNet, and NotSimpleNet, with 1,665,835 parameters, 2,102,347 parameters, and 11,803,003 parameters respectively. I trained all my models locally on an RTX 4090. The train-validation-test split was 0.8-0.15-0.05 throughout the process.
## Modifications to the provided code
The provided CircleDetection.py file was modified to support PyTorch's autograd, without which the model would not have trained, as the loss function would not be differentiable otherwise. 
This involved replacing the numpy vectors with PyTorch vectors. 
Furthermore, to use it as a loss metric, I used 1-IOU, as Intersection over Union is an accuracy measure, not a loss function.
(I realised this the hard way after training a SimpleNet on it and having circles completely outside the bounds).

## Networks
### SimpleNet
SimpleNet was trained over a dataset of 10,000 examples (split as aforementioned), for 100 training steps. It was trained using SGD on a learning rate of 0.01 and a momentum of 0.9.

![image](https://github.com/ved07/SlingshotAI-Circle-Detection-Challenge/assets/49959052/beca337b-4b57-4f47-9a60-74963d1a72d3)

It performed well, but began to overfit as the training progressed, and did not have a particularly high accuracy over the validation data. 

### NotSoSimpleNet
NotSoSimpleNet was trained over a dataset of 20,000 examples (split as aforementioned), for 100 training steps. It was trained using SGD on a learning rate of 0.01 and a momentum of 0.9.

![image](https://github.com/ved07/SlingshotAI-Circle-Detection-Challenge/assets/49959052/592bb4a4-8770-4445-b39b-9b42fb3d2b19)

It performed similar to SimpleNet, but did not overfit to the same extent, as neither loss had plateaued by the end of training. I decided to up the parameter count based on what was said in the question document. I also decided to change the optimiser to ADAM as I wanted faster convergence.

### NotSimpleNet
NotSimpleNet was the final iteration, due to this model being ~5x bigger than the previous network, it took a significantly longer amount of time to train locally. It was also trained on 5 times as many examples, with a total dataset size of 100,000
This model had to go through some hyperparameter tuning, specifically reducing the learning rate to 0.001, as a few iterations got stuck at a local minima.

As aforementioned, NotSimpleNet was trained over a dataset of 100,000 examples (split as aforementioned), for 100 training steps. It was trained using Adam on a learning rate of 0.001.
It performed significantly better than the other models, however the increase in parameters largely originated from a "beefier" DNN block. Using more convolutions would probably have been more effective, for the same parameter count.

![image](https://github.com/ved07/SlingshotAI-Circle-Detection-Challenge/assets/49959052/ddd0edbb-deeb-4176-82f0-12ff15719524)

## Final Remarks
Overall, this project involved a non-negligible amount of debugging, due to my choice of using PyTorch, rather than a simpler framework such as Keras; all the things that make PyTorch great actually hindered me, as they don't lend themselves as well to simpler tasks.
