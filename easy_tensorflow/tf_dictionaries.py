import tensorflow as tf

transform_dict = {
    'relu': (tf.nn.relu, (tf.nn.bias_add, ((tf.matmul, ('X', 'weight1')), 'weight2'))),
    'softplus': (tf.nn.softplus, (tf.nn.bias_add, ((tf.matmul, ('X', 'weight1')), 'weight2'))),
    'dropout': (tf.nn.dropout, ('X', 'weight')),
    'bias_add': (tf.nn.bias_add, ('X', 'weight')),
    'sigmoid': (tf.nn.sigmoid, (tf.nn.bias_add, ((tf.matmul, ('X', 'weight1')), 'weight2'))),
    'tanh': (tf.nn.tanh, (tf.nn.bias_add, ((tf.matmul, ('X', 'weight1')), 'weight2'))),
    'none': (tf.nn.bias_add, ((tf.matmul, ('X', 'weight1')), 'weight2')),
    'normalize': (tf.nn.l2_normalize, ('X', 1)),
    'sum': (tf.reduce_sum, ('X', 1)),
    'prod': (tf.reduce_prod, ('X', 1)),
    'min': (tf.reduce_min, ('X', 1)),
    'max': (tf.reduce_max, ('X', 1)),
    'mean': (tf.reduce_mean, ('X', 1)),
    'softmax': (tf.nn.softmax, (tf.matmul, ('X', 'weight')))
}

optimizer_dict = {
    'GradientDescent': (tf.train.GradientDescentOptimizer, (0.001, )),
    'Adagrad': (tf.train.AdagradOptimizer, (0.001, )),
    'Momentum': (tf.train.MomentumOptimizer, (0.001, 0.1)),
    'Adam': (tf.train.AdamOptimizer, ()),
    'Ftrl': (tf.train.FtrlOptimizer, (0.001, )),
    'RMSProp': (tf.train.RMSPropOptimizer, (0.001, 0.9))
}
