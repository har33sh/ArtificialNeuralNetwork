# Neural Network
- Implemenation of Neural Network
- Done in Python, Pandas and Numpy with no external machine learning used
<br />
The purpose of this project was to understand the architecture of Neural Network and to know what is going on during the training process.<br />

**This is not a production quality code**
<br />

## About the Implemenation
 Implemenation is inspired by the [MLPClassifier of sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)  

### Configurable Parameters
- **hidden_layer_size**:
    The number of hidden layersm by default the size of hidden layer is set to be 3, as in most of the cases 3 layers is good.
- **learning_rate**: The rate at which the  weights are updated
- **neurons**: The number of neurons in the hidden layers
- **activation_function**
  - tanh: the hyperbolic tan function, returns f(x) = tanh(x). *This is the default*
  - relu: the rectified linear unit function, returns f(x) = max(0, x)
  - sigmoid: the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).

- **iterations**: Maximum number of iterations.
- **decay_factor**:Should be between 0 and 1. The rate at which the learning_rate rate is decayed


## Problem Statement
(https://inclass.kaggle.com/c/cs-725-403-assignment-2)

Task is to predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset.
<br />
Note that in the train and test data,salary >$50K is represented by 1 and <=$50K is represented by 0.
<br />

To know more about the dataset [click here](https://har33sh.github.io/ArtificialNeuralNetwork/)
<br />


## References
1. [Census Income Data Set from archive.ics.uci.edu ](https://archive.ics.uci.edu/ml/datasets/Census+Income)
2. [A guide to Deep learning](http://yerevann.com/a-guide-to-deep-learning/)
3. [Information on how to optimise Neural Network](http://cs231n.github.io/neural-networks-3/)
4. [Neural Network in 11 lines -- Short and best Implemenation of NN ](http://iamtrask.github.io/2015/07/12/basic-python-network/)
