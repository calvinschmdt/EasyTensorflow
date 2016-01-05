import numpy as np
import scipy.stats
import random
from deap import base
from deap import creator
from deap import tools
import tf_functions

def rmse(p, t):
    '''
    Calculates the root means squared error between two vectors.
    :param p: One-dimensional numpy array with the predicted values.
    :param t: One-dimensional numpy array with the known values.
    :return: Float value between 0 and infinity.
    '''

    return np.sqrt(np.average(np.square(np.subtract(p, t))))

def r_squared(p, t):
    '''
    Coefficient of determination for determining how well the data fits the model.
    :param p: Numpy array with the predicted values.
    :param t: Numpy array with the known values.
    :return: Float value between 0 and 1.
    '''

    # Reshapes into one-dimensional array if necessary.
    if len(t.shape) == 2:
        t = [i[0] for i in t]

    if len(p.shape) == 2:
        p = [i[0] for i in p]

    # Get r value for squaring.
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(p, t)
    return r_value ** 2

def accuracy(p, t):
    '''
    Fraction of samples that were classified correctly.
    :param p: Multi-dimensional numpy array with the predicted values.
    :param t: Multi-dimensional numpy array with the known values.
    :return: Float value between 0 and 1.
    '''

    return sum([1 for i, j in zip(np.argmax(p, axis=1), np.argmax(t, axis=1)) if i == j]) / float(len(p))

def specificity(p, t):
    '''
    Fraction of positive samples correctly identified as positive.
    :param p: Multi-dimensional numpy array with the predicted values.
    :param t: Multi-dimensional numpy array with the known values.
    :return: Float value between 0 and 1.
    '''

    # Calculates number of correctly identified positive samples.
    num = sum([1 for i, j in zip(np.argmax(p, axis=1), np.argmax(t, axis=1)) if i == 0 and j == 0])

    # Calculates total number of positive samples.
    denom = sum([1 for i in np.argmax(t, axis=1) if i == 0])

    return num / float(denom)

def sensitivity(p, t):
    '''
    Fraction of negative samples correctly identified as negative.
    :param p: Multi-dimensional numpy array with the predicted values.
    :param t: Multi-dimensional numpy array with the known values.
    :return: Float value between 0 and 1.
    '''

    # Calculates number of correctly identified negative samples.
    num = sum([1 for i, j in zip(np.argmax(p, axis=1), np.argmax(t, axis=1)) if i == 1 and j == 1])

    # Calculates total number of negative samples.
    denom = sum([1 for i in np.argmax(t, axis=1) if i == 1])

    return num / float(denom)

def test_train_split(X, y, num_test = 0):
    '''
    Splits a full set of input and output data randomly into train and test sets. Keeps the input and output values
    connected.
    :param X: Numpy array with input data.
    :param y: Numpy array with output data.
    :param num_test: Number of samples to use for the test set. Set to 0 by default, which cause 1/5 of the full set to
        be split off for testing.
    :return: Four numpy arrays corresponding to the training input, training output, testing input, and testing output
        data.
    '''

    # Splits off 1/5 of data if no specific amount is given.
    if num_test == 0:
        num_test = y.shape[0] / 5

    # Turns one-dimensional vector into array of shape (n, 1).
    if len(y.shape) == 1:
        y = y.reshape(len(y), 1)

    # Records the number of output features.
    output_features = y.shape[1]

    # Combines the input and output features so that the outputs can stay connected.
    all_vals = np.append(X, y, 1)

    # Randomly shuffles the samples.
    np.random.shuffle(all_vals)

    # Pulls out the test and train input and output features.
    teX, teY = all_vals[:num_test, :-output_features], all_vals[:num_test, -output_features:]
    trX, trY = all_vals[num_test:, :-output_features], all_vals[num_test:, -output_features:]

    return trX, trY, teX, teY

def evolve(predict_type, fitness_measure, trX, trY,
           max_layers = 5, num_gens = 10, gen_size = 40, teX = [], teY = [],
           layer_types = ['relu', 'softplus', 'dropout', 'bias_add', 'sigmoid', 'tanh', 'none', 'normalize'],
           layer_sizes = [0, 0, 10, 50, 100, 200, 500, 1000],
           end_types = ['sum', 'prod', 'min', 'max', 'mean', 'none', 'sigmoid', 'tanh'],
           train_types = ['GradientDescent', 'GradientDescent', 'GradientDescent', 'Adagrad', 'Momentum', 'Adam', 'Ftrl', 'RMSProp'],
           cross_prob = 0.2, mut_prob = 0.2, tourn_size = 5, train_iters = 5, squash_errors = True):
    '''

    :param predict_type: String denoting the type of neural network to evolve. Two options: 'regression' and
        'classification'.
    :param fitness_measure: String denoting the type of measurement to use for evaluating the performance of the network
        type. Options:
		- 'rmse': Root mean squared error between the predicted values and known values. Use for regression.
		- 'r_squared': Coefficient of determination for determining how well the data fits the model. Use for
		    regression.
		- 'accuracy': Fraction of samples that were classified correctly. Use for classification, and can be used for
		    multi-class classification.
		- 'sensitivity': Fraction of positive samples correctly identified as positive. Use for classification with two
		    classes, and the second class is the positive class.
		- 'specificity': Fraction of negative samples correctly identified as negative. Use for classification with two
		    classes, and the first class is the negative class.
    :param trX: Numpy array with input data to use for training. Will pull randomly from this array to create test and
        training sets.
    :param trY: Numpy array with output data to use for training.
    :param max_layers: Integer denoting the maximum number of layers that exist between the input and output layer. Set
        at 5 by default.
    :param num_gens: Number of generations to simulate. Set at 10 by default.
    :param gen_size: Number of individual members per generation. Set at 40 by default.
    :param teX: If a specific test set is desired, enter the input data here as a numpy array.
    :param teY: Test output data as a numpy array.
    :param layer_types: List of strings denoting the layer types possible to be used. Set to ['relu', 'softplus',
        'dropout', 'bias_add', 'sigmoid', 'tanh', 'none', 'normalize'] by default.
    :param layer_sizes: List of integers denoting the layer sizes possible to be used. List must be of the same length
        as layer_types. Set to [0, 0, 10, 50, 100, 200, 500, 1000] by default.
    :param end_types: List of strings denoting the options for the type of transformation that gives the output. List
        must be of same length as layer_types. Set to ['sum', 'prod', 'min', 'max', 'mean', 'none', 'sigmoid', 'tanh']
        by default.
    :param train_types: List of strings denoting the optimizer types possible to be used. List must be of same length as
        layer_types. Set to ['GradientDescent', 'GradientDescent', 'GradientDescent', 'Adagrad', 'Momentum', 'Adam',
        'Ftrl', 'RMSProp'] by default.
    :param cross_prob: Float value denoting the probability of crossing the genetics of different individuals. Set at
        0.2 by default.
    :param mut_prob: Float value denoting the probability of changing the genetics of a single individual. Set at 0.2 by
        default.
    :param tourn_size: Integer denoting the number of individuals to carry from each generation. Set at 5 by default.
    :param train_iters: Integer denoting the number of training iterations to use for each neural network. Set at 5 by
        default.
    :param squash_errors: Boolean value denoting whether to give a fail value if the network results in an error. Set to
        True by default.
    :return: List of strings giving the best net_type, string denoting the best optimizer, and Float value denoting the
        measure of the best network type.
    '''

    # Checks that the different options have the same size.
    if not len(layer_types) == len(layer_sizes) == len(end_types) == len(train_types):
        print('Input attribute lists have different sizes.')
        return None

    # Gets the type of network to check.
    if predict_type == 'regression':
        predictor = tf_functions.Regresser
    elif predict_type == 'classification':
        predictor = tf_functions.Classifier
        end_types = ['softmax'] * len(layer_types)

    # Gets the type of success measure to use.
    if fitness_measure == 'rmse':
        measure = rmse
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        fail_val = np.inf
    elif fitness_measure == 'r_squared':
        measure = r_squared
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        fail_val = 0
    elif fitness_measure == 'accuracy':
        measure = accuracy
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        fail_val = 0
    elif fitness_measure == 'specificity':
        measure = specificity
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        fail_val = 0
    elif fitness_measure == 'sensitivity':
        measure = sensitivity
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        fail_val = 0

    toolbox = base.Toolbox()

    # Attribute generator.
    toolbox.register("attr_ints", random.randint, 0, len(layer_types) - 1)

    # Structure initializers.
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_ints, n=(max_layers * 2) + 2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operator registering.
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt,low = 0, up = len(layer_types) - 1, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize = tourn_size)

    # Gets the initial population.
    pop = toolbox.population(n = gen_size)

    # Performs an initial selection.
    for ind in pop:

        # Turns the individual's genes into a net_type and an optimizer.
        net_type = []
        for i in range(max_layers):
            net_type.append(layer_types[ind[i * 2]])
            net_type.append(layer_sizes[ind[i * 2 + 1]])

        net_type.append(end_types[ind[-2]])

        # Splits into test and train if needed.
        if teX == []:
            ttrX, ttrY, tteX, tteY = test_train_split(trX, trY)
        else:
            ttrX, ttrY, tteX, tteY = trX, trY, teX, teY

        # Attempts to test network if errors to be squashed.
        if squash_errors:
            try:

                # Sets up, trains, and tests network.
                ind_predictor = predictor(net_type, optimizer = train_types[ind[-1]])
                ind_predictor.train(ttrX, ttrY, train_iters)

                p = ind_predictor.predict(tteX)
                m = measure(p, tteY)

                ind_predictor.close()

            # Upon an error, gives the worst possible value.
            except:
                m = fail_val

            if np.isnan(m):
                m = fail_val

        else:
            ind_predictor = predictor(net_type, optimizer = train_types[ind[-1]])
            ind_predictor.train(ttrX, ttrY, train_iters)

            p = ind_predictor.predict(tteX)
            m = measure(p, tteY)

            ind_predictor.close()

        ind.fitness.values = (m,)

    # Begins the evolution.
    for g in range(num_gens):

        # Selects the next generation individuals.
        offspring = toolbox.select(pop, len(pop))
        # Clones the selected individuals.
        offspring = list(map(toolbox.clone, offspring))

        # Applies crossover and mutation on the offspring.
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cross_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mut_prob:

                toolbox.mutate(mutant)

                del mutant.fitness.values

        # Evaluates the individuals with an invalid fitness.
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            net_type = []
            for i in range(max_layers):
                net_type.append(layer_types[ind[i * 2]])
                net_type.append(layer_sizes[ind[i * 2 + 1]])

            net_type.append(end_types[ind[-2]])

            if squash_errors:
                try:
                    ind_predictor = predictor(net_type, optimizer = train_types[ind[-1]])
                    ind_predictor.train(ttrX, ttrY, train_iters)

                    p = ind_predictor.predict(tteX)
                    m = measure(p, tteY)

                    ind_predictor.close()

                except:
                    m = fail_val

                if np.isnan(m):
                    m = fail_val

            else:
                ind_predictor = predictor(net_type, optimizer = train_types[ind[-1]])
                ind_predictor.train(ttrX, ttrY, train_iters)

                p = ind_predictor.predict(tteX)
                m = measure(p, tteY)

                ind_predictor.close()

            ind.fitness.values = (m,)

        # The population is entirely replaced by the offspring.
        pop[:] = offspring

    # Gets the best individual remaining after
    best_ind = tools.selBest(pop, 1)[0]

    net_type = []
    for i in range(max_layers):
        net_type.append(layer_types[best_ind[i * 2]])
        net_type.append(layer_sizes[best_ind[i * 2 + 1]])

    net_type.append(end_types[best_ind[-2]])
    optimizer = train_types[best_ind[-1]]

    return net_type, optimizer, best_ind.fitness.values
