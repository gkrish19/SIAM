import numpy as np
import tensorflow as tf

WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'

FLAGS1 = tf.app.flags.FLAGS

def _get_split_q(ngroups, dim, name='split', l2_loss=False):  #Gives a tesnor of size ngroups*dim with constant rando values with std dev of 0.001.
    with tf.variable_scope(name):

        std_dev = 0.001
        init_val = np.random.normal(0, std_dev, (ngroups, dim)) #0 is the centre, std_dev the standard dev and then the last term the Output shape.
                                                                #If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
        print('Init value is:', init_val)
        init_val = init_val - np.average(init_val, axis=0) + 1.0/ngroups
        print('New Init value is:', init_val)
        q = tf.get_variable('q', shape=[ngroups, dim], dtype=tf.float32,
                                initializer=tf.constant_initializer(init_val))
        print('Return value is:',q)

    return q

def _merge_split_q(q, merge_idxs, name='merge'):  #Returns a list which is splits it as the maxvalue to the value at location dimesnsion size
                                                  #and then appends the set to that.
    assert len(q.get_shape()) == 2
    ngroups, dim = q.get_shape().as_list()
    print ('The ngroups and dim are :', ngroups, dim)
    assert ngroups == len(merge_idxs)

    with tf.variable_scope(name):
        max_idx = np.max(merge_idxs)
        temp_list = []
        for i in range(max_idx + 1):
            temp = []
            for j in range(ngroups):
                if merge_idxs[j] == i:
                    temp.append(tf.slice(q, [j, 0], [1, dim]))
            temp_list.append(tf.add_n(temp))
        ret = tf.concat(temp_list, 0)

    return ret

def _add_flops_weights(scope_name, f, w):

	#if scope_name not in _counted_scope:
	FLAGS1._flops = 0
	FLAGS1._weights = 0
	FLAGS1._flops += f
	FLAGS1._total_flops += FLAGS1._flops
	FLAGS1._weights += w
	FLAGS1._total_weights += FLAGS1._weights
	print("\nThe number of flops for layer %s is:%g\n" %(scope_name, (FLAGS1._flops)))
	print("\n The number of weights for layer %s is: %g \n" %(scope_name, (FLAGS1._weights)))



def _get_even_merge_idxs(N, split):  #Returns the expanded value for the split and nothing else
    assert N >= split   #Tree Structure Can we change this ?
    num_elems = [(N + split - i - 1)/split for i in range(split)]
    expand_split = [[i] * n for i, n in enumerate(num_elems)]
    return [t for l in expand_split for t in l]

def _fc_1(x, out_dim, input_q=None, output_q=None, name="fc"):
    print("Entered _fc_1\n\n")
    b, in_dim = x.get_shape().as_list()
    x = _fc(x, out_dim, input_q, output_q, name)
    f = 2 * (in_dim + 1) * out_dim
    w = (in_dim + 1) * out_dim
    scope_name = tf.get_variable_scope().name + "/" + name
    #_add_flops_weights(scope_name, f, w)
    return x

def _fc(x, out_dim, input_q=None, output_q=None, name='fc'):
    if (input_q == None)^(output_q == None):
        raise ValueError('Input/Output splits are not correctly given.')

    with tf.variable_scope(name):
        # Main operation: fc

        w = tf.get_variable('weights', [x.get_shape()[1], out_dim],
                        tf.float32, initializer=tf.random_normal_initializer(
                            stddev=np.sqrt(1.0/x.get_shape().as_list()[1])))
        b = tf.get_variable('biases', [out_dim], tf.float32,
                            initializer=tf.constant_initializer(0.0))
        if w not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, w)
            # print('\tadded to WEIGHT_DECAY_KEY: %s(%s)' % (w.name, str(w.get_shape().as_list())))

        fc = tf.nn.bias_add(tf.matmul(x, w), b)
        #fc = tf.matmul(x, w), b)

        # Split loss
        if (input_q is not None) and (output_q is not None):
            print('Adding the split loss now as both the splits are none')
            print('input_q:' ,input_q)
            _add_split_loss(w, input_q, output_q)

    return fc

def _conv(x, filter_size, out_channel, strides, pad='SAME', input_q=None, output_q=None, name='conv'):
    if (input_q == None)^(output_q == None):
        raise ValueError('Input/Output splits are not correctly given.')

    in_shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        # Main operation: conv2d
        with tf.device('/CPU:0'):
            kernel = tf.get_variable('kernel', [filter_size, filter_size, in_shape[3], out_channel],
                            tf.float32, initializer=tf.random_normal_initializer(
                                stddev=np.sqrt(1.0/filter_size/filter_size/in_shape[3])))
        if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
            # print('\tadded to WEIGHT_DECAY_KEY: %s(%s)' % (kernel.name, str(kernel.get_shape().as_list())))
        conv = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], pad)

        # Split and split loss
        if (input_q is not None) and (output_q is not None):
            _add_split_loss(kernel, input_q, output_q)

    return conv







def _add_split_loss(w, input_q, output_q):
    # Check input tensors' measurements
    assert len(w.get_shape()) == 2 or len(w.get_shape()) == 4
    in_dim, out_dim = w.get_shape().as_list()[-2:] #Need to understand
    print('The input dimension is:', in_dim)
    print('The output dimension is:', out_dim)
    print('Input_q is',input_q.get_shape().as_list()[1])
    print('OUtput_q is',output_q.get_shape().as_list()[1])
    assert len(input_q.get_shape()) == 2
    assert len(output_q.get_shape()) == 2
    assert in_dim == input_q.get_shape().as_list()[1]
    assert out_dim == output_q.get_shape().as_list()[3]
    assert input_q.get_shape().as_list()[0] == output_q.get_shape().as_list()[0]  # ngroups
    ngroups = input_q.get_shape().as_list()[0]
    assert ngroups > 1

    # Add split losses to collections
    T_list = []
    U_list = []
    if input_q not in tf.get_collection('OVERLAP_LOSS_WEIGHTS') \
            and not "concat" in input_q.op.name:
        tf.add_to_collection('OVERLAP_LOSS_WEIGHTS', input_q)
        print('\t\tAdd overlap & split loss for %s' % input_q.name)
        T_temp, U_temp = ([], []) # U_temp holds the squares of the sum and T_temp holds the sum of the squares
        for i in range(ngroups):
            for j in range(ngroups):
                if i <= j:
                    continue
                T_temp.append(tf.reduce_sum(input_q[i,:] * input_q[j,:]))
            U_temp.append(tf.square(tf.reduce_sum(input_q[i,:])))
        T_list.append(tf.reduce_sum(T_temp)/(float(in_dim*(ngroups-1))/float(2*ngroups))) #the denominator is a normalisation process
        U_list.append(tf.reduce_sum(U_temp)/(float(in_dim*in_dim)/float(ngroups))) #again the denominator is a constant value and we are scaling the numerator
    if output_q not in tf.get_collection('OVERLAP_LOSS_WEIGHTS') \
            and not "concat" in output_q.op.name:
        print('\t\tAdd overlap & split loss for %s' % output_q.name)
        tf.add_to_collections('OVERLAP_LOSS_WEIGHTS', output_q)
        T_temp, U_temp = ([], [])
        for i in range(ngroups):
            for j in range(ngroups):
                if i <= j:
                    continue
                T_temp.append(tf.reduce_sum(output_q[i,:] * output_q[j,:]))
            U_temp.append(tf.square(tf.reduce_sum(output_q[i,:])))
        T_list.append(tf.reduce_sum(T_temp)/(float(out_dim*(ngroups-1))/float(2*ngroups)))
        U_list.append(tf.reduce_sum(U_temp)/(float(out_dim*out_dim)/float(ngroups)))
    if T_list:
        tf.add_to_collection('OVERLAP_LOSS', tf.add_n(T_list)/len(T_list))
    if U_list:
        tf.add_to_collection('UNIFORM_LOSS', tf.add_n(U_list)/len(U_list))

    S_list = []  #This holds the value of Group Weight Regularisation
    if w not in tf.get_collection('WEIGHT_SPLIT_WEIGHTS'):
        tf.add_to_collection('WEIGHT_SPLIT_WEIGHTS', w)

        ones_col = tf.ones((in_dim,), dtype=tf.float32)
        ones_row = tf.ones((out_dim,), dtype=tf.float32)
        print ('\nThe shape of w is:\n', len(w.get_shape()))
        if len(w.get_shape()) == 4:
            w_reduce = tf.reduce_mean(tf.square(w), [0, 1])
            w_norm = w_reduce
            std_dev = np.sqrt(1.0/float(w.get_shape().as_list()[0])**2/in_dim)
            # w_norm = w_reduce / tf.reduce_sum(w_reduce)
        else:
            w_norm = w #For the FC Layer only
            std_dev = np.sqrt(1.0/float(in_dim))
            # w_norm = w / tf.sqrt(tf.reduce_sum(tf.square(w)))

        for i in range(ngroups):
            if len(w.get_shape()) == 4:
                wg_row = tf.transpose(tf.transpose(w_norm * tf.square(output_q[i,:])) * tf.square(ones_col - input_q[i,:])) #THis holds W*Q*(I-P)
                wg_row_l2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(wg_row, 1))) / (in_dim*np.sqrt(out_dim))
                wg_col = tf.transpose(tf.transpose(w_norm * tf.square(ones_row - output_q[i,:])) * tf.square(input_q[i,:]))
                wg_col_l2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(wg_col, 0))) / (np.sqrt(in_dim)*out_dim)
            else:  # len(w.get_shape()) == 2
                wg_row = tf.transpose(tf.transpose(w_norm * output_q[i,:]) * (ones_col - input_q[i,:]))
                wg_row_l2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(wg_row * wg_row, 1))) / (in_dim*np.sqrt(out_dim))#HEere they do L21 regularisation, need to read the paper.
                wg_col = tf.transpose(tf.transpose(w_norm * (ones_row - output_q[i,:])) * input_q[i,:])
                wg_col_l2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(wg_col * wg_col, 0))) / (np.sqrt(in_dim)*out_dim)  #HEere they do L21 regularisation, need to read the paper.
            S_list.append(wg_row_l2 + wg_col_l2)
        # S = tf.add_n(S_list)/((ngroups-1)/ngroups)
        S = tf.add_n(S_list)/(2*(ngroups-1)*std_dev/ngroups)  #This divides the sum by a threshold. What is the need???
        tf.add_to_collection('WEIGHT_SPLIT', S)

        print('Weight split loss is:', tf.get_collection('WEIGHT_SPLIT'))

        # Add histogram for w if split losses are added
        scope_name = tf.get_variable_scope().name
        print(tf.summary.histogram("%s/" % scope_name, w))
        print('\t\tAdd split loss for %s(%dx%d, %d groups)' \
            % (tf.get_variable_scope().name, in_dim, out_dim, ngroups))

    return




############################################################################################
############################################################################################
"""
Reload of Weights
"""
def get_sparse_fc2(directory):
    reader = tf.train.NewCheckpointReader(FLAGS1.train_dir)
    Weights = reader.get_tensor(directory)
    wei = Weights
    shape = Weights.shape
    no_nodes = shape[0] + shape[1]
    Weights_mean = np.mean(Weights)
    Weights_std = np.std(Weights)
    threshold1 = Weights_mean - FLAGS1.beta1*Weights_std
    threshold2 = Weights_mean + FLAGS1.beta1*Weights_std
    print("Thresholds 1 and 2 are", threshold1, threshold2)
    name = directory

    ones = 0
    zeros =0
    w_sparse = np.zeros(shape=shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if((Weights[i][j]<= threshold1) & (Weights[i][j]>= threshold2)):
                ones = ones+1
                w_sparse[i][j] = 1

            else:
                zeros =zeros + 1
                Weights[i][j] = 0

    sparsity = (float(zeros) / float(shape[0] * shape[1]))*100
    sparsity_1 = (float(ones) / float(shape[0] * shape[1]))
    f = 2*(shape[0] + 1) * shape[1]
    w = ((shape[0] + 1) * shape[1])
    scope_name = tf.get_variable_scope().name + "/" + name
    _add_flops_weights(scope_name, f, w)
    FLAGS1._total_flops_sw = FLAGS1._total_flops*(sparsity_1)
    FLAGS1._total_weights_sw = FLAGS1._total_weights*(sparsity_1)

    print("Sparse matrix is", w_sparse)

    print ("The std deviation matrix is", Weights_std)
    print("The number of ones and zeros are %d %d" %(ones,zeros))
    print("Sparsity of %s is %g percentage" %(name, sparsity))
    print("\nThe number of flops in SW is:%g\n" %((FLAGS1._total_flops_sw)))
    print("\n The number of weights in SW is: %g \n" %((FLAGS1._total_weights_sw)))

    print("Weight Matrix that is sparse and not binary is", Weights)

    return(Weights, FLAGS1._total_flops_sw, FLAGS1._total_weights_sw)


def get_sparse_prob(directory):
    reader = tf.train.NewCheckpointReader(FLAGS1.train_dir)
    Weights = reader.get_tensor(directory)
    wei = Weights
    shape = Weights.shape
    no_nodes = shape[0] + shape[1]
    Weights_mean = np.mean(Weights)
    Weights_std = np.std(Weights)
    threshold1 = Weights_mean - FLAGS1.beta1*Weights_std
    threshold2 = Weights_mean + FLAGS1.beta1*Weights_std
    print("Thresholds 1 and 2 are", threshold1, threshold2)
    name = directory

    ones = 0
    zeros =0
    w_sparse = np.zeros(shape=shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if((Weights[i][j]>= threshold1) & (Weights[i][j]<= threshold2)):
                ones = ones+1
                w_sparse[i][j] = 1

            else:
                zeros =zeros + 1
                Weights[i][j] = 0

    sparsity = (float(zeros) / float(shape[0] * shape[1]))*100
    sparsity_1 = (float(ones) / float(shape[0] * shape[1]))
    f = 2*(shape[0] + 1) * shape[1]
    w = ((shape[0] + 1) * shape[1])
    scope_name = tf.get_variable_scope().name + "/" + name
    _add_flops_weights(scope_name, f, w)
    FLAGS1._total_flops_sw = FLAGS1._total_flops*(sparsity_1)
    FLAGS1._total_weights_sw = FLAGS1._total_weights*(sparsity_1)

    print("Sparse matrix is", w_sparse)

    print ("The std deviation matrix is", Weights_std)
    print("The number of ones and zeros are %d %d" %(ones,zeros))
    print("Sparsity of %s is %g percentage" %(name, sparsity))
    print("\nThe number of flops in SW is:%g\n" %((FLAGS1._total_flops_sw)))
    print("\n The number of weights in SW is: %g \n" %((FLAGS1._total_weights_sw)))

    print("Weight Matrix that is sparse and not binary is", Weights)

    return(Weights, FLAGS1._total_flops_sw, FLAGS1._total_weights_sw)

def get_sparse_sw_fc(directory, Weights, beta):
    #reader = tf.train.NewCheckpointReader(FLAGS1.train_dir)
    print("Sparsity Percentage set for layer %s is %g" %(directory,beta))
    #Weights = reader.get_tensor(directory)
    wei1 = np.absolute(Weights)
    shape = wei1.shape
    no_nodes = shape[0] * shape[1]
    wei = np.ndarray.flatten(wei1)
    shape1 = wei.shape
    print(shape1)
    max = np.max(wei)
    weights_sort = np.sort(wei)
    no_ele = weights_sort.size
    print(no_ele)
    no_pruned = int(float(no_ele * (1-beta)))
    print("The number of elements prunded for a percent of %g are %d" %(((1-beta)*100), no_pruned))
    if (no_pruned ==0):
        wei_pruned = Weights
    elif(no_pruned ==no_nodes):
        wei_pruned = np.zeros(shape=shape)
    else:
        ele = no_pruned
        threshold = weights_sort[ele]
        print("Threshold value is %g" %(threshold))
        mask=np.zeros(shape=shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if(wei1[i][j]>threshold):
                    mask[i][j] = 1
        wei_pruned = np.multiply(Weights,mask)
    name = directory
    ones = 0
    zeros =0
    ones = np.count_nonzero(wei_pruned)
    zeros =no_nodes - ones
    sparsity = (float(zeros) / float(shape[0] * shape[1]))*100
    sparsity_1 = (float(ones) / float(shape[0] * shape[1]))
    f = 2*(shape[0] + 1) * shape[1]
    w = ((shape[0] + 1) * shape[1])
    scope_name = tf.get_variable_scope().name + "/" + name
    _add_flops_weights(scope_name, f, w)
    FLAGS1._total_flops_sw = FLAGS1._flops*(sparsity_1)
    FLAGS1._total_weights_sw = FLAGS1._weights*(sparsity_1)

    #print("Sparse matrix is", wei_pruned)
    print("The number of ones and zeros are %d %d" %(ones,zeros))
    print("Sparsity of %s is %g percentage" %(name, sparsity))
    print("\nThe number of flops in SW is:%g\n" %((FLAGS1._total_flops_sw)))
    print("\n The number of weights in SW is: %g \n" %((FLAGS1._total_weights_sw)))
    wei_pruned = tf.convert_to_tensor(wei_pruned, dtype=tf.float32)
    return(wei_pruned)


def get_sparse_fc(directory, beta):
    reader = tf.train.NewCheckpointReader(FLAGS1.train_dir)
    print("Sparsity Percentage set for layer %s is %g" %(directory,beta))
    Weights = reader.get_tensor(directory)
    wei1 = np.absolute(Weights)
    shape = wei1.shape
    no_nodes = shape[0] * shape[1]
    wei = np.ndarray.flatten(wei1)
    shape1 = wei.shape
    max = np.max(wei)
    weights_sort = np.sort(wei)
    no_ele = weights_sort.size
    print(no_ele)
    no_pruned = int(float(no_ele * (1-beta)))
    print("The number of elements prunded for a percent of %g are %d" %(((1-beta)*100), no_pruned))
    if (no_pruned ==0):
        wei_pruned = Weights
    elif(no_pruned ==no_nodes):
        wei_pruned = np.zeros(shape=shape)
    else:
        ele = no_pruned
        threshold = weights_sort[ele]
        print("Threshold value is %g" %(threshold))
        mask=np.zeros(shape=shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if(wei1[i][j]>threshold):
                    mask[i][j] = 1
        wei_pruned = np.multiply(Weights,mask)
    name = directory
    ones = 0
    zeros =0
    ones = np.count_nonzero(wei_pruned)
    zeros =no_nodes - ones
    sparsity = (float(zeros) / float(shape[0] * shape[1]))*100
    sparsity_1 = (float(ones) / float(shape[0] * shape[1]))
    f = 2*(shape[0] + 1) * shape[1]
    w = ((shape[0] + 1) * shape[1])
    scope_name = tf.get_variable_scope().name + "/" + name
    _add_flops_weights(scope_name, f, w)
    FLAGS1._total_flops_sw = FLAGS1._flops*(sparsity_1)
    FLAGS1._total_weights_sw = FLAGS1._weights*(sparsity_1)

    #print("Sparse matrix is", wei_pruned)
    print("The number of ones and zeros are %d %d" %(ones,zeros))
    print("Sparsity of %s is %g percentage" %(name, sparsity))
    print("\nThe number of flops in SW is:%g\n" %((FLAGS1._total_flops_sw)))
    print("\n The number of weights in SW is: %g \n" %((FLAGS1._total_weights_sw)))
    return(wei_pruned, FLAGS1._total_flops_sw, FLAGS1._total_weights_sw)

def get_sparse_conv(directory, beta):
    reader = tf.train.NewCheckpointReader(FLAGS1.train_dir)
    print("Sparsity Percentage set for layer %s is %g" %(directory,beta))
    Weights = reader.get_tensor(directory)
    wei1 = np.absolute(Weights)
    shape = wei1.shape
    no_nodes = shape[0] * shape[1] * shape[2] * shape[3]
    wei = np.ndarray.flatten(wei1)
    shape1 = wei.shape
    max = np.max(wei)
    weights_sort = np.sort(wei)
    no_ele = weights_sort.size
    no_pruned = int(float(no_ele * (1-beta)))
    print("The number of elements prunded for a percent of %g are %d" %(((1-beta)*100), no_pruned))
    if (no_pruned ==0):
        wei_pruned = Weights
    elif(no_pruned ==no_nodes):
        wei_pruned = np.zeros(shape=shape)
    else:
        ele = no_pruned
        threshold = weights_sort[ele]
        print("Threshold value is %g" %(threshold))
        mask=np.zeros(shape=shape)
        for i in range(shape[3]):
            for j in range(shape[2]):
                for k in range(shape[1]):
                    for l in range(shape[0]):
                        if(wei1[l][k][j][i]>threshold):
                            mask[l][k][j][i] = 1
        wei_pruned = np.multiply(Weights,mask)
    name = directory
    ones = 0
    zeros =0
    ones = np.count_nonzero(wei_pruned)
    zeros =no_nodes - ones
    sparsity = (float(zeros) / float(no_nodes))*100
    sparsity_1 = (float(ones) / float(no_nodes))
    value = tf.convert_to_tensor(Weights, dtype=tf.float32)
    b, h, w,in_channel = value.get_shape().as_list()
    f = 2 * (h) * (w) * in_channel * shape[3] * shape[0] * shape[1]
    w = (in_channel * shape[3] * shape[0] * shape[1])
    scope_name = tf.get_variable_scope().name + "/" + name
    _add_flops_weights(scope_name, f, w)
    FLAGS1._total_flops_sw = FLAGS1._flops*(sparsity_1)
    FLAGS1._total_weights_sw = FLAGS1._weights*(sparsity_1)

    #print("Sparse matrix is", wei_pruned)
    print("The number of ones and zeros are %d %d" %(ones,zeros))
    print("Sparsity of %s is %g percentage" %(name, sparsity))
    print("\nThe number of flops in SW is:%g\n" %((FLAGS1._total_flops_sw)))
    print("\n The number of weights in SW is: %g \n" %((FLAGS1._total_weights_sw)))
    return(wei_pruned, FLAGS1._total_flops_sw, FLAGS1._total_weights_sw)

def get_sparse_sw_conv(directory, Weights, beta):
    #reader = tf.train.NewCheckpointReader(FLAGS1.train_dir)
    print("Sparsity Percentage set for layer %s is %g" %(directory,beta))
    #Weights = reader.get_tensor(directory)
    wei1 = np.absolute(Weights)
    shape = wei1.shape
    no_nodes = shape[0] * shape[1] * shape[2] * shape[3]
    wei = np.ndarray.flatten(wei1)
    shape1 = wei.shape
    max = np.max(wei)
    weights_sort = np.sort(wei)
    no_ele = weights_sort.size
    no_pruned = int(float(no_ele * (1-beta)))
    print("The number of elements prunded for a percent of %g are %d" %(((1-beta)*100), no_pruned))
    if (no_pruned ==0):
        wei_pruned = Weights
    elif(no_pruned ==no_nodes):
        wei_pruned = np.zeros(shape=shape)
    else:
        ele = no_pruned
        threshold = weights_sort[ele]
        print("Threshold value is %g" %(threshold))
        mask=np.zeros(shape=shape)
        for i in range(shape[3]):
            for j in range(shape[2]):
                for k in range(shape[1]):
                    for l in range(shape[0]):
                        if(wei1[l][k][j][i]>threshold):
                            mask[l][k][j][i] = 1
        wei_pruned = np.multiply(Weights,mask)
    name = directory
    ones = 0
    zeros =0
    ones = np.count_nonzero(wei_pruned)
    zeros =no_nodes - ones
    sparsity = (float(zeros) / float(no_nodes))*100
    sparsity_1 = (float(ones) / float(no_nodes))
    value = tf.convert_to_tensor(Weights, dtype=tf.float32)
    b, h, w,in_channel = value.get_shape().as_list()
    f = 2 * (h) * (w) * in_channel * shape[3] * shape[0] * shape[1]
    w = (in_channel * shape[3] * shape[0] * shape[1])
    scope_name = tf.get_variable_scope().name + "/" + name
    _add_flops_weights(scope_name, f, w)
    FLAGS1._total_flops_sw = FLAGS1._flops*(sparsity_1)
    FLAGS1._total_weights_sw = FLAGS1._weights*(sparsity_1)

    #print("Sparse matrix is", wei_pruned)
    print("The number of ones and zeros are %d %d" %(ones,zeros))
    print("Sparsity of %s is %g percentage" %(name, sparsity))
    print("\nThe number of flops in SW is:%g\n" %((FLAGS1._total_flops_sw)))
    print("\n The number of weights in SW is: %g \n" %((FLAGS1._total_weights_sw)))
    wei_pruned = tf.convert_to_tensor(wei_pruned, dtype=tf.float32)
    return(wei_pruned)

def get_sparse_base(directory):
    reader = tf.train.NewCheckpointReader(FLAGS1.train_dir)
    Weights = reader.get_tensor(directory)
    shape = Weights.shape


    return(Weights)
