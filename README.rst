=========================
 Easy Tensorflow
=========================

This package provides users with methods for the automated building, training, and testing of complex neural networks using Google's Tensorflow package. The project includes objects that perform both regression and classification tasks.

In addition, there is a function included that uses the DEAP genetic algorithm package to evolve the optimal network architecture. The evolution function is almost entirely based off of the sample DEAP evolution.

This project is meant to simplify the tensorflow experience, and therefore it reduces the customizibility of the networks. Patches that expand functionality are welcome and encouraged, as long they do not reduce the simplicity of usage. I will try to keep up with maintenance as best as I can, but please be patient; I am new to this.

Project Setup
=============

Dependancies
------------

Full support for Python 2.7. Python 3.3 not tested.

Requires tensorflow (tested on version 0.6.0). Installation instructions `on the Tensorflow website <https://www.tensorflow.org/versions/master/get_started/os_setup.html>`_ .

Requires DEAP (tested on version 1.0) for evolving. Installation instructions `here <http://deap.readthedocs.org/en/1.0.x/installation.html>`_.

Installation
------------

1. Either download and extract the zipped file, or clone directly from github using::

    git clone https://github.com/calvinschmdt/EasyTensorflow.git easy_tensorflow

   This should create a new directory containing the required files.
    
2. Install the dependancies manually or by running this command while in the easy_tensorflow directory::

    sudo pip install -r requirements.txt

3. Install the project by running this command while in the easy_tensorflow directory::

    sudo python setup.py install
    
Usage
=====

Prediction Objects
------------------

This package uses objects to hold the neural networks. There are separate objects for performing regression and classification (the two objects are Regresser and Classifier), but the two objects have the same basic functions.

Instantiation
-------------

Instantiate the object by assigning it to a variable. The only required argument for instantiation is a list that describes the neural network::

    net_type = ['none', 20, 'sigmoid', 30, 'bias_add', 30, 'sigmoid']
    reg = etf.tf_functions.Regresser(net_type)
      
The net_type list needs to be in a specific format: alternating strings and integers starting and ending with a string. The strings describe the transformation that is made between each layer of the neural network, while the integers denote the number of size of the layer after the transformation is made. For example, the above network would look like this:

    +---------------------------------------------------------------------------+
    |                  Input with n samples and f features                      |
    +---------------------------------------------------------------------------+
    |              *Matrix multiplication by adjustable weights*                |
    +---------------------------------------------------------------------------+
    |                 Layer 1 with n samples and 20 features                    |
    +---------------------------------------------------------------------------+
    | *Sigmoid transformation on a matrix multiplication by adjustable weights* |
    +---------------------------------------------------------------------------+
    |                 Layer 2 with n samples and 30 features                    |
    +---------------------------------------------------------------------------+
    |            *Addition to each feature of an adjustable weight*             |
    +---------------------------------------------------------------------------+
    |                 Layer 3 with n samples and 30 features                    |
    +---------------------------------------------------------------------------+
    | *Sigmoid transformation on a matrix multiplication by adjustable weights* |
    +---------------------------------------------------------------------------+
    |                  Output with n samples and o features                     |
    +---------------------------------------------------------------------------+


The number of input and output features do not have to be specified upon instantiation, but are learned during training.

The transformations available are:
    
    - 'relu': Relu transformation on a matrix multiplication by adjustable weights. Relu applies a ramp function.
    - 'softplus': Softplus transformation on a matrix multiplication by adjustable weights. Softplus applies a smoothed ramp function.
    - 'dropout': Randomly pushes features to zero. Prevents overfitting. Does not change the number of features.
    - 'bias_add': Adds an adjustable weight to each feature. Does not change the number of features.
    - 'sigmoid': Sigmoid transformation on a matrix multiplication by adjustable weights. Sigmoid forces values to approach 0 or 1.
    - 'tanh': Hyperbolic tangent transformation on a matrix multiplication by adjustable weights. Tanh forces values very positive or very negative.
    - 'none': Matrix multiplication with adjustable weights.
    - 'normalize': Normalizes features of a sample using an L2 norm. Does not change the number of features.
    - 'sum': Sum across all features. Reduces to 1 feature, so most useful as a final transformation in a regression.
    - 'prod': Multiplies all features. Reduces to 1 feature, so most useful as a final transformation in a regression.
    - 'min': Takes minimum value of all features. Reduces to 1 feature, so most useful as a final transformation in a regression.
    - 'max': Takes maximum value of all features. Reduces to 1 feature, so most useful as a final transformation in a regression.
    - 'mean': Takes mean of all features. Reduces to 1 feature, so most useful as a final transformation in a regression.
    - 'softmax': Normalizes the features so that the sum equals 1 on a matrix multiplication by adjustable weights. Most useful as a final transformation in a classification to give class probabilities.

The object has several optional arguments:

    loss_type: String that defines the error measurement term. This is used during training to determine the weights that give the most accurate output. The loss types available are:
        
        - 'l2_loss' - Uses tensorflow's nn.l2_loss function on the difference between the predicted and actual. Computes half the L2 norm without the sqrt. Use for regression, and the default loss_type for the regression object.
        - 'cross_entropy' - Calculates the cross-entropy between two probability distributions as defined in Tensorflow's MNIST tutorial (-tf.reduce_sum(y * tf.log(py_x))). Use for classification, and the default loss_type for the classification object.
        
    optimizer: String that defines the optimization algorithm for training. If a string is passed, the optimizers will be used with default learning rates. If you wish to use a custom training rate, instead of a string, pass in a tuple with the tensorflow optimizer as the first index, and a tuple with the arguments to pass in as the second index. The optimizers available are:
        
        - 'GradientDescent': Implements the gradient descent algorithm with a default learning rate of 0.001.
        - 'Adagrad': Implements the Adagrad algorithm with a default learning rate of 0.001.
        - 'Momentum': Implements the Momentum algorithm with a default learning rate of 0.001 and momentum of 0.1.
        - 'Adam': Implements the Adam algorithm.
        - 'FTRL': Implements the FTRL algorithm with a learning rate of 0.001.
        - 'RMSProp': Implements the RMSProp algorithm with a learning rate of 0.001 and a decay of 0.9.
        
Training
--------

Objects are trained by calling object.train() with certain arguments::

    trX = training_data
    try = training_output
    training_steps = 50
    reg.train(trX, try, training_steps)

Both objects are trained by passing in a set of data with known outputs. The training input data should be passed in as a numpy array, with each sample as a row and features as the columns. The training output data can take multiple forms: 

    - For regression tasks, it can be an iterable list with one output value for each sample, or it can be a numpy array of shape (n, 1).
    - For classification tasks, it can be a numpy array of shape (n, m), where m is the number of classes. In this array, there is a 1 in each row in the column of the class that that sample belongs to, and a 0 in all other rows. Otherwise, an iterable list can be passed in with the class name for each sample. This is required is the class names, and not a probability matrix, are to be returned during testing.
    
In addition to the training data and training output, the number of times to iterate over training must be passed in as the third argument.

There are several optional arguments for training that control how long training the network takes:

    - full_train: Denotes whether to use the entire training set each iteration. Set to True by default.
    - train_size: If full train is set to False, denotes how many samples to use from the training set each iteration of training. Pulls randomly from the training set with possible repeats.
    
Predicting
----------

After the object is trained, the network can be used to predict the output of test data that is given to it by calling object.predict() with certain arguments::

    teX = test_data
    p = reg.predict(teX)
      
The test data should have the same number of features as the training data, though the number of samples may be different.

The output for a regression object will be a numpy array of shape (n, ) with the predicted value for each sample.

The output for a classification object will be a list with a predicted class for each sample. If a probability matrix is desired, the pass the argument return_encoded = False when predicting, and a numpy array of shape (n, m) will be returned.

Closing
-------

Calling object.close() will close the network, freeing up resources. It cannot be used again, and a new object must be started for training and predicting to occur.

Evolving
========

For those who do not know the neural network architecture for your problem, we can use a genetic selection algorithm to evolve the optimal architecture.

To do this, use the command evolve() with several required arguments:

    - predict_type: String denoting the type of neural network to evolve. Two options: 'regression' and 'classification'.
    - fitness_measure: String denoting the type of measurement to use for evaluating the performance of the network type. Options:
        - 'rmse': Root mean squared error between the predicted values and known values. Use for regression.
        - 'r_squared': Coefficient of determination for determining how well the data fits the model. Use for regression.
        - 'accuracy': Fraction of samples that were classified correctly. Use for classification, and can be used for multi-class classification.
        - 'sensitivity': Fraction of positive samples correctly identified as positive. Use for classification with two classes, and the second class is the positive class.
        - 'specificity': Fraction of negative samples correctly identified as negative. Use for classification with two classes, and the first class is the negative class.
    - trX: Numpy array with input data to use for training. Will pull randomly from this array to create test and training sets.
    - trY: Numpy array with output data to use for training.

After the evolution finishes, it will return a net_type and optimizer that can be fed into an regression or classification object, along with the measurement that net_type produced. If "Error during training" is printed, it only means that an error was encountered at some point during the evolution.::

    net_type, opt, m = etf.evolve_functions.evolve('classification', 'accuracy', trX, trY)

There are many optional arguments that allow for customization of the evolution:

	- max_layers: Integer denoting the maximum number of layers that exist between the input and output layer. Set at 5 by default.
	- num_gens: Number of generations to simulate. Set at 10 by default.
	- gen_size: Number of individual members per generation. Set at 40 by default.
	- teX: If a specific test set is desired, enter the input data here as a numpy array.
	- teY: Test output data as a numpy array.
	- layer_types: List of strings denoting the layer types possible to be used. Can have repeated types for an increased probability of incorporation. Set to ['relu', 'softplus', 'dropout', 'bias_add', 'sigmoid', 'tanh', 'none', 'normalize'] by default.
	- layer_sizes: List of integers denoting the layer sizes possible to be used. Layer sizes of 0 drop out a layer. List must be of the same length as layer_types. Set to [0, 0, 10, 50, 100, 200, 500, 1000] by default.
	- end_types: List of strings denoting the options for the type of transformation that gives the output. Forced to be softmax by default during classification. List must be of same length as layer_types. Set to ['sum', 'prod', 'min', 'max', 'mean', 'none', 'sigmoid', 'tanh'] by default.
	- train_types: List of strings denoting the optimizer types possible to be used. List must be of same length as layer_types. Set to ['GradientDescent', 'GradientDescent', 'GradientDescent', 'Adagrad', 'Momentum', 'Adam', 'Ftrl', 'RMSProp'] by default.
	- cross_prob: Float value denoting the probability of crossing the genetics of different individuals. Set at 0.2 by default.
	- mut_prob: Float value denoting the probability of changing the genetics of a single individual. Set at 0.2 by default.
	- tourn_size: Integer denoting the number of individuals to carry from each generation. Set at 5 by default.
	- train_iters: Integer denoting the number of training iterations to use for each neural network. Set at 5 by default.
	- squash_errors: Boolean value denoting whether to give a fail value if the network results in an error. Set to True by default. Recommended to leave true, as it is difficult to complete a long evolution without running into some type of error.

Licenses
========

The code which makes up this Python project template is licensed under the MIT/X11 license. Feel free to use it in your free software/open-source or proprietary projects.

Issues
======

Please report any bugs or requests that you have using the GitHub issue tracker!

Development
===========

If you wish to contribute, first make your changes. Then run the following from the project root directory::

    source internal/test.sh

This will copy the template directory to a temporary directory, run the generation, then run tox. Any arguments passed will go directly to the tox command line, e.g.::

    source internal/test.sh -e py27

This command line would just test Python 2.7.

Acknowledgements
================

Both Tensorflow and DEAP were creating by other (very smart) people, this package just combines the two.

This package was set up using Sean Fisk's Python Project Template package.

Authors
=======

* Calvin Schmidt
