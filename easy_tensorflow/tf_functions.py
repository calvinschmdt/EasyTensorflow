import tensorflow as tf
import numpy as np
import random
import pickle
import tf_dictionaries

def random_index_list(list_size, sample_size):
    '''
    Creates a list of random integers that are constrained in the max value. May have repeating indexes.
    :param list_size: Integer denoting the length of the final list.
    :param sample_size: Integer denoting the maximum index to include.
    :return: List of integers.
    '''

    return [random.randint(0, sample_size - 1) for i in range(0, list_size)]

def encode_classifications(class_list):
    '''
    Given a list of class labels, encodes the list in a manner that is usable in neural networks. For each sample,
    there will be a list of 0s and one 1 corresponding to the label that sample is encoded. Also returns the list of
    so that encoded sample lists can be decoded.
    :param class_list: List of labels, that can be either strings or numbers.
    :return: classes: List of strings denoting the different labels found in the list.
        encoded: Numpy array of encoded labels. One row for each sample, with the numbers in the row corresponding to
            the label for that sample.
    '''

    # Iterates through the list, recording all the samples.
    classes = []
    for i in class_list:
        if i not in classes:
            classes.append(i)

    # Creates a list of 0s for each sample.
    encoded = [[0] * len(classes) for i in range(len(class_list))]

    # Iterates through the list to be encoded, marking a 1 in the position for the given label for that sample.
    for e, i in enumerate(class_list):
        encoded[e][classes.index(i)] = 1

    return classes, np.array(encoded)

def decode_classifications(classes, encoded):
    '''
    Given an array of labels encoded in array format, returns a that array as their original labels.
    :param classes: List of strings denoting the different labels found in the list.
    :param encoded: Numpy array of encoded labels. One row for each sample, with the numbers in the row corresponding to
        the label for that sample.
    :return: List of strings denoting the label of each sample.
    '''

    return [classes[i] for i in encoded]

def unpack_transform(transform, X, weight):
    '''
    Turns a tuple of tensorflow functions and their inputs into tensorflow functions ready to use.
    :param transform: Tuple containing a tensorflow tensor as the first part, and a tuple containing the tensor input in
        the second part.
    :param X: Tensor containing the input matrix to be transformed.
    :param weight: Tensor, integer, or float containing a weight as the second tensor input.
    :return: Completed tensor, with inputs placed in.
    '''

    # If the first part of the tensor's input should be X, then applies the X as the incoming tensor.
    if transform[1][0] == 'X':

        # If a weight tensor or value should be applied to the tensor, then uses the supplied weight. If there is not a
        # weight, uses 1 to direct the axis of tensor transformation.
        if transform[1][1] == 'weight':
            w = weight
        elif transform[1][1] == 1:
            w = 1

        return transform[0](X, w)

    # If the first part of the tensor's input is not X, then unpacks the tensor that should be the input, and uses that.
    else:
        return transform[0](unpack_transform(transform[1], X, weight))

def compile_model(X, weights, models, transform_dict):
    '''
    Converts lists of weights and models into a tensorflow neural network that can take in an input tensor and return
    an output tensor.
    :param X: Tensor that contains the input values to be transformed through the network.
    :param weights: List of tensors (or single value in the case of dropouts) that transform the X value through the
        layers.
    :param models: List of the different types of layers. Should be in the same order as the weights list, and be the
        desired order of layers.
    :return: Tensor that has been transformed through the different layers.
    '''

    # Iterates through each layer.
    for model, weight in zip(models, weights):

        # Finds the layer type, and transforms through that layer type.
        transform = transform_dict[model]

        X = unpack_transform(transform, X, weight)

    return X

def make_model(X, input_size, output_size, net_type, transform_dict):
    '''
    Creates the lists of weight tensors and layer types that are used to compile a tensorflow neural network.
    :param X: Tensor that contains the input values to be transformed through the network.
    :param input_size: Float value giving the number of features being fed into the network.
    :param output_size: Float value giving the number of output values desired.
    :param net_type: List of alternating string values and integer values. Must always start and end with a string
        values. The strings denote the type of each layer. The integer values denote the end size of each layer, though
        this is constrained for certain layer types. Sizes of zero drop that layer out.
    :return: Function that represents the neural network.
    '''

    weights = []
    models = []
    started = False
    last = input_size

    # Iterates through each layer of the network.
    for e in range(0, len(net_type) - 1, 2):

        # Creates the weight type for the layer, depending on the layer type. Dropouts don't use a tensor, just a float
        # value.
        if net_type[e] == 'dropout' or net_type[e] == 'normalize':
            next = last
            weight = random.random()

        # Bias add uses a linear layer to transform without changing the size.
        elif net_type[e] == 'bias_add':
            next = last
            weight = tf.Variable(tf.constant(0.0, shape=[next]))

        # Other types use a matrix to transform the size.
        else:
            next = net_type[e + 1]
            weight = tf.Variable(tf.random_normal([last, next], stddev=0.01))

        # Records the layer type and weight used for transforming into that layer for each layer.
        if net_type[e + 1] > 0:

            # The first layer pulls from the input size, while the rest pull from the last size.
            if not started:
                weights.append(weight)
                models.append(net_type[e])
                last = next
                started = True
            else:
                weights.append(weight)
                models.append(net_type[e])
                last = next

    # Makes sure the last layer can change the size.
    if net_type[-1] == 'bias_add' or net_type[-1] == 'dropout' or net_type[-1] == 'normalize':
        final = 'sigmoid'
    else:
        final = net_type[-1]

    # Adds in the last layer.
    weights.append(tf.Variable(tf.random_normal([last, output_size], stddev=0.01)))
    models.append(final)

    # Returns the compiled function.
    return compile_model(X, weights, models, transform_dict)

def train_tensorflow(sess, trX, trY, train_steps, full_train, train_size, net_type, transform_dict, loss_type,
                     optimizer, optimizer_dict):
    '''
    Automatically constructs, trains, and tests a tensorflow neural network, returning the r squared value of the
    output.
    :param sess: A tensorflow session.
    :param trX: Numpy array that contains the training features.
    :param trY: Numpy array that contains the training outputs. Must have shape of at least 1 on columns.
    :param train_steps: Integer value denoting the number of times to iterate through training.
    :param full_train: Boolean value denoting whether to use the full training set for each iteration.
    :param train_size: Integer value denoting the number of samples to pull from the training set for each iteration of
        training.
    :param net_type: List of alternating string values and integer values. Must always start and end with a string
        values. The strings denote the type of each layer. The integer values denote the end size of each layer, though
        this is constrained for certain layer types. Sizes of zero drop that layer out.
    :param transform_dict: Dictionary of strings to tuples of tensors that encode how to set up the layers of the neural
        network.
    :param loss_type: String denoting type of tensor to use for loss type. Use l2_loss for regression, cross_entropy for
        classification.
    :param optimizer: String denoting the type of optimization tensor to use for training the neural network.
    :param optimizer_dict: Dictionary of strings to tuples of tensors that encode how to set up the optimizers of the
        neural network.
    :return: predict_op: Tensor that encodes the neural network.
        X: Placeholder tensor for the features array.
        y: Placeholder tensor for the output array.
    '''

    # Set up input and output tensors.
    X = tf.placeholder("float", [None, trX.shape[1]])
    y = tf.placeholder("float", [None, trY.shape[1]])

    # Set up network.
    py_x = make_model(X, trX.shape[1], trY.shape[1], net_type, transform_dict)

    # Set up cost and training type.
    if loss_type == 'l2_loss':
        cost = tf.nn.l2_loss(tf.sub(py_x, y))
    elif loss_type == 'cross_entropy':
        cost = -tf.reduce_sum(y * tf.log(py_x))

    # Gets the optimizer to be used for training and sets it up.
    if type(optimizer) == str:
        train_op = optimizer_dict[optimizer][0](*optimizer_dict[optimizer][1]).minimize(cost)
    else:
        train_op = optimizer[0](*optimizer[1]).minimize(cost)

    predict_op = py_x

    # Initialize session.
    init = tf.initialize_all_variables()
    sess.run(init)

    # Trains given number of times
    try:
        for i in range(train_steps):

            # If full_train is selected, the trains on the full set of training data, in 100 sample increments.
            if full_train:
                for start, end in zip(range(0, len(trX), 100), range(100, len(trX), 100)):
                    sess.run(train_op, feed_dict={X: trX[start:end], y: trY[start:end]})

            # If full_train is not selected, then trains on a random set of samples from the training data.
            else:
                indices = random_index_list(train_size, len(trY))
                sess.run(train_op, feed_dict={X: trX[indices], y: trY[indices]})

    # If training throws an error for whatever reason, stops the program from breaking. Throws an error message and \
    # closes the session.
    except:
        print("Error during training")
        sess.close()
        return None

    return predict_op, X, y

class Classifier:
    '''
    Object that holds a neural network used for classifying a set of data.
    '''

    def __init__(self, net_type, loss_type = 'cross_entropy', optimizer = 'Adam'):
        '''
        Initializing function. Records the neural network type and, if given, the loss type and optimizer type.
        :param net_type: List of alternating string values and integer values. Must always start and end with a string
        values. The strings denote the type of each layer. The integer values denote the end size of each layer, though
        this is constrained for certain layer types. Sizes of zero drop that layer out.
        :param loss_type: String denoting type of tensor to use for loss type. Set as cross_entropy by default.
        :param optimizer: String denoting the type of optimization tensor to use for training the neural network. Set
            as Adam by default. Tuple containing optimization tensor and input tuple can be used in place of a string to
            set specific parameters for the optimization.
        '''

        # Sets up initial parameters and starts the tensorflow session.
        self.net_type = net_type
        self.loss_type = loss_type
        self.optimizer = optimizer
        self.sess = tf.Session()
        self.transform_dict = tf_dictionaries.transform_dict
        self.optimizer_dict = tf_dictionaries.optimizer_dict

    def train(self, trX, trY, iterations, full_train = True, train_size = 0):
        '''
        Sets up and trains the neural network.
        :param trX: Numpy array that contains the training features.
        :param trY: Numpy array that contains the training outputs. Can be in the encoded or unencoded format.
        :param iterations: Integer denoting the number of iterations to train the model.
        :param full_train: Boolean value denoting whether to use the full training set for each iteration. Set as True
            by default.
        :param train_size: Integer value denoting the number of samples to pull from the training set for each iteration
            of training. Set as 0 by default.
        :return: Does not return anything, but stores the input, output, and transformation tensors for predicting.
        '''

        # If the labels are not encoded in matrix format, does that and stores the list of classes.
        if trY.shape[1] == 1:
            self.class_list, trY = encode_classifications(trY)

        # Sets up and trains tensorflow.
        self.predict_op, self.X, self.y = train_tensorflow(self.sess, trX, trY, iterations,
                                                           full_train, train_size,
                                                           self.net_type, self.transform_dict,
                                                           self.loss_type,
                                                           self.optimizer, self.optimizer_dict)

    def predict(self, teX, return_encoded = True):
        '''
        Uses the trained neural network to classify the samples based on their features.
        :param teX: Numpy array of features to be used for the classification.
        :param return_encoded: Boolean value denoting whether to decode the classifications if needed. Must have
            generated class list by encoding during the training.
        :return: Either encoded or decoded classifications for the input samples as a numpy array or list.
        '''

        # Use the neural network for predicting the classes.
        p = self.sess.run(self.predict_op, feed_dict={self.X: teX})

        # Decodes if desired.
        if not return_encoded:
            p = np.argmax(p, axis=1)
            return decode_classifications(p, self.class_list)

        return p

    def close(self):
        '''
        Closes the session.
        :return: Nothing, but closes the session.
        '''

        self.sess.close()

class Regresser:
    '''
    Object that holds a neural network used for predicting a set of data's numerical outputs.
    '''

    def __init__(self, net_type, loss_type = 'l2_loss', optimizer = 'Adam'):
        '''
        Initializing function. Records the neural network type and, if given, the loss type and optimizer type.
        :param net_type: List of alternating string values and integer values. Must always start and end with a string
        values. The strings denote the type of each layer. The integer values denote the end size of each layer, though
        this is constrained for certain layer types. Sizes of zero drop that layer out.
        :param loss_type: String denoting type of tensor to use for loss type. Set as l2_loss by default.
        :param optimizer: String denoting the type of optimization tensor to use for training the neural network. Set
            as Adam by default. Tuple containing optimization tensor and input tuple can be used in place of a string to
            set specific parameters for the optimization.
        '''

        # Sets up initial parameters and starts the tensorflow session.
        self.net_type = net_type
        self.loss_type = loss_type
        self.optimizer = optimizer
        self.sess = tf.Session()
        self.transform_dict = tf_dictionaries.transform_dict
        self.optimizer_dict = tf_dictionaries.optimizer_dict

    def train(self, trX, trY, iterations, full_train = True, train_size = 0):
        '''
        Sets up and trains the neural network.
        :param trX: Numpy array that contains the training features.
        :param trY: Numpy array that contains the training outputs. If if in vector format, then reshapes as array.
        :param iterations: Integer denoting the number of iterations to train the model.
        :param full_train: Boolean value denoting whether to use the full training set for each iteration. Set as True
            by default.
        :param train_size: Integer value denoting the number of samples to pull from the training set for each iteration
            of training. Set as 0 by default.
        :return: Does not return anything, but stores the input, output, and transformation tensors for predicting.
        '''

        # The training outputs are not in the correct shape, fits into correct shape.
        if len(trY.shape) == 1:
            trY = trY.reshape(len(trY), 1)

        # Sets up and trains tensorflow.
        self.predict_op, self.X, self.y = train_tensorflow(self.sess, trX, trY, iterations,
                                                           full_train, train_size,
                                                           self.net_type, self.transform_dict,
                                                           self.loss_type,
                                                           self.optimizer, self.optimizer_dict)

    def predict(self, teX):
        '''
        Uses the trained neural network to classify the samples based on their features.
        :param teX: Numpy array of features to be used for the classification.
        :return: Numpy vector of predicted outputs..
        '''

        # Use the neural network for predicting the classes.
        p = self.sess.run(self.predict_op, feed_dict={self.X: teX})

        # Reshapes array of columns into vector.
        if len(p.shape) > 1:
            p = np.array([i[0] for i in np.ndarray.tolist(p)])

        return np.array(p)

    def close(self):
        '''
        Closes the session.
        :return: Nothing, but closes the session.
        '''

        self.sess.close()
