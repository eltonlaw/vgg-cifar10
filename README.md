# VGG

TensorFlow implementation of the model described in "Very Deep Convolutional Networks for Large Scale Image Recognition"

Model adapted for CIFAR-10.

### Running the script

1. Clone this repository
		
	```bash
	$ git clone https://github.com/eltonlaw/machine-learning-models.git	$ cd machine-learning-models/VGG
	```
2. Download the dataset. Run the following from command line or if you're not comfortable with that, manually download the CIFAR-10 dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html) and move the tar file to `../machine-learning-models/VGG/`
	
	```bash
	$ curl -o cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
	```
		
3. Run the model
 	
 	```bash
	$ python3 vgg.py
	```
		
	Default parameters are the ones I found to work best after fine-tuning. To change, them just pass values through the command line.

	* `learning_rate`
	* `batch_size`
	* `epochs`

	```bash
	$ python3 vgg.py --learning_rate 1e-5 --batch_size=256 --epochs 100
	```	
		
### About this Implementation

Note: The original paper was performed on scaled-down ImageNet images (following the AlexNet architecture). I first experimented with scaling each image to (224,224,3) using the original parameters from the paper. This was followed by a round of attempts at fine-tuning these parameters. Another experiment was ran on the original, non-scaled images in the same 'spirit' of the original paper (using small filters and deep architecture). To see the results, look below. 

#### Accessing the Data

The dataset used is CIFAR-10, which consists of 60 000 32x32 RGB images in 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. 

After unpickling, you get a 10,000 x 3072 numpy array. The images have been flattened into a 1D 3072 vector. The standard format for images is 32x32x3 so I reshaped each 1D vector and plotted it.  

```python3
...
img = np.reshape(img, [32, 32, 3])
...
```

![](https://github.com/eltonlaw/vgg-cifar10/blob/master/images/data_preprocessing_1.png?raw=true)

The data is tiled, but luckily someone on [StackOverflow knows](https://stackoverflow.com/questions/28005669/how-to-view-an-rgb-image-with-pylab). Basically, it has to do with the order in which the data is reshaped. The default for a numpy reshape is `C` which means to read/write elements in C-like index order. Using a Fortran-like index order, `F` will solve the problem.

```python3
...
img = np.reshape(img, [32, 32, 3], order="F")
...
```

![](https://github.com/eltonlaw/vgg-cifar10/blob/master/images/data_preprocessing_2.png?raw=true)

Awesome, they actually look like images now. For some reason everything's rotated 90 degrees counterclockwise. This won't affect classification accuracy so it's not a big problem unless we want to view the images or do transfer learning. Let's say we do (it's not too hard anyways). 

```python3
...
img = np.reshape(img, [32, 32, 3], order="F")
img = np.rot90(img, k=3)
...
```

![](https://github.com/eltonlaw/vgg-cifar10/blob/master/images/data_preprocessing_3.png?raw=true)

Perfect. The labels correspond correctly and everything else looks fine. We can move on to the machine learning now.

#### Data Preprocessing

Data preprocessing consists of just a standardization step. 

> "The only pre- processing we do is subtracting the mean RGB value, computed on the training set, from each pixel."



### References

A. Krizhevsky. Learning multiple layers of features from tiny images. Master’s thesis, Department of Computer Science, University of Toronto, 2009.

A. Krizhevsky. cuda-convnet. https://code.google.com/p/cuda-convnet/, 2012.

A. Krizhevsky, I. Sutskever, and G. Hinton. ImageNet classification with deep convolutional neural networks. In NIPS, 2012.

I. Sutskever, J. Martens, G. E. Dahl, and G. E. Hinton. On the importance of initialization and momentum in deep learning. In ICML, volume 28 of JMLR Proceedings, pages 1139–1147. JMLR.org, 2013.

N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, pages 1929–1958, 2014.

V. Nair and G. Hinton. Rectified linear units improve restricted boltzmann machines. In Proc. 27th  International Conference on Machine Learning, 2010. 

Y. Boureau, J. Ponce, and Y. LeCun. A Theoretical Analysis of Feature Pooling in Visual Recognition. In International Conference on Machine Learning, 2010.
