import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
from tensorflow.python.ops import rnn
from utils import pytorch_kaiming_weight_factor

##################################################################################
# Initialization
##################################################################################

# factor, mode, uniform = pytorch_kaiming_weight_factor(a=0.0, uniform=False)
# weight_init = tf_contrib.layers.variance_scaling_initializer(factor=factor, mode=mode, uniform=uniform)
weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)

weight_regularizer = None
weight_regularizer_fully = None

##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x


def fully_connected(x, units, use_bias=True, sn=False, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn:
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                initializer=weight_init, regularizer=weight_regularizer_fully)
            if use_bias:
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))

        else:
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                                kernel_regularizer=weight_regularizer_fully,
                                use_bias=use_bias)

        return x


def flatten(x):
    return tf.layers.flatten(x)

def various_rnn(x, n_layer=1, n_hidden=128, dropout_rate=0.5, bidirectional=True, rnn_type='lstm', scope='rnn') :

    if rnn_type.lower() == 'lstm' :
        cell_type = tf.nn.rnn_cell.LSTMCell
    elif rnn_type.lower() == 'gru' :
        cell_type = tf.nn.rnn_cell.GRUCell
    else :
        raise NotImplementedError

    with tf.variable_scope(scope):
        if bidirectional:
            if n_layer > 1 :
                fw_cells = [cell_type(n_hidden) for _ in range(n_layer)]
                bw_cells = [cell_type(n_hidden) for _ in range(n_layer)]

                if dropout_rate > 0.0:
                    fw_cell = [tf.nn.rnn_cell.DropoutWrapper(cell=fw_cell, output_keep_prob=1 - dropout_rate) for fw_cell in fw_cells]
                    bw_cell = [tf.nn.rnn_cell.DropoutWrapper(cell=bw_cell, output_keep_prob=1 - dropout_rate) for bw_cell in bw_cells]

                fw_cell = tf.nn.rnn_cell.MultiRNNCell(fw_cell)
                bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cell)

            else :
                fw_cell = cell_type(n_hidden)
                bw_cell = cell_type(n_hidden)

                if dropout_rate > 0.0 :
                    fw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=fw_cell, output_keep_prob=1 - dropout_rate)
                    bw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=bw_cell, output_keep_prob=1 - dropout_rate)

            outputs, states = rnn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=x, dtype=tf.float32)
            # outputs = 모든 state
            # states = 마지막 state = output[-1]
            output_fw, output_bw = outputs[0], outputs[1] # [bs, seq_len, n_hidden]
            state_fw, state_bw = states[0], states[1]

            words_emb = tf.concat([output_fw, output_bw], axis=-1) # [bs, seq_len, n_hidden * 2]

            # state_fw[0] = cell state
            # state_fw[1] = hidden state

            if rnn_type.lower() == 'lstm':
                sent_emb = tf.concat([state_fw[1], state_bw[1]], axis=-1) # [bs, n_hidden * 2]
            elif rnn_type.lower() == 'gru':
                sent_emb = tf.concat([state_fw, state_bw], axis=-1)  # [bs, n_hidden * 2]
            else :
                raise NotImplementedError

        else :
            if n_layer > 1 :
                cells = [cell_type(n_hidden) for _ in range(n_layer)]

                if dropout_rate > 0.0 :
                    cell = [tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=1 - dropout_rate) for cell in cells]

                cell = tf.nn.rnn_cell.MultiRNNCell(cell)
            else :
                cell = cell_type(n_hidden)

                if dropout_rate > 0.0 :
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=1 - dropout_rate)

            outputs, states = rnn.dynamic_rnn(cell, inputs=x, dtype=tf.float32)

            words_emb = outputs # [bs, seq_len, n_hidden]

            # states[0] = cell state
            # states[1] = hidden state
            if rnn_type.lower() == 'lstm' :
                sent_emb = states[1] # [bs, n_hidden]
            elif rnn_type.lower() == 'gru' :
                sent_emb = states # [bs, n_hidden]
            else :
                raise NotImplementedError

        return words_emb, sent_emb


##################################################################################
# Residual-block
##################################################################################


def resblock(x_init, channels, is_training=True, use_bias=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels * 2, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = batch_norm(x, is_training)
            x = glu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = batch_norm(x, is_training)

        return x + x_init

def up_block(x_init, channels, is_training=True, use_bias=True, sn=False, scope='up_block'):
    with tf.variable_scope(scope):
        x = up_sample(x_init, scale_factor=2)
        x = conv(x, channels * 2, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
        x = batch_norm(x, is_training)
        x = glu(x)

        return x

def down_block(x_init, channels, is_training=True, use_bias=True, sn=False, scope='down_block'):
    with tf.variable_scope(scope):
        x = conv(x_init, channels, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
        x = batch_norm(x, is_training)
        x = lrelu(x, 0.2)

        return x


def attention_net(x, word_emb, mask, channels, use_bias=True, sn=False, scope='attention_net'):
    with tf.variable_scope(scope):
        bs, h, w = x.shape[0], x.shape[1], x.shape[2]
        hw = h * w # length of query
        seq_len = word_emb.shape[1] # length of source

        x = tf.reshape(x, shape=[bs, hw, -1])
        word_emb = tf.expand_dims(word_emb, axis=1)
        word_emb = conv(word_emb, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv')
        word_emb = tf.squeeze(word_emb, axis=1)


        attn = tf.matmul(x, word_emb, transpose_b=True) # [bs, hw, seq_len]
        attn = tf.reshape(attn, shape=[bs * hw, seq_len])


        mask = tf.tile(mask, multiples=[hw, 1])
        attn = tf.where(tf.equal(mask, True), x=tf.constant(-float('inf'), dtype=tf.float32, shape=mask.shape), y=attn)
        attn = tf.nn.softmax(attn)

        attn = tf.reshape(attn, shape=[bs, hw, seq_len])

        weighted_context = tf.matmul(word_emb, attn, transpose_a=True, transpose_b=True)
        weighted_context = tf.reshape(weighted_context, shape=[bs, h, w, -1])
        attn = tf.reshape(attn, shape=[bs, h, w, -1])

        return weighted_context, attn
##################################################################################
# Sampling
##################################################################################

def dropout(x, drop_rate=0.5, is_training=True):
    return tf.layers.dropout(x, drop_rate, training=is_training)

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def resize(x, target_size):
    return tf.image.resize_bilinear(x, size=target_size)

def down_sample_avg(x, scale_factor=2):
    return tf.layers.average_pooling2d(x, pool_size=3, strides=scale_factor, padding='SAME')

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap

def reparametrize(mean, logvar):
    eps = tf.random_normal(tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32)

    return mean + tf.exp(logvar * 0.5) * eps

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

def simoid(x) :
    return tf.sigmoid(x)

def glu(x) :
    ch = x.shape[-1]
    ch = ch // 2

    n_dim = len(np.shape(x))

    if n_dim == 2:
        return x[:, :ch] * simoid(x[:, ch:])

    else : # n_dim = 4
        return x[:, :, :, :ch] * simoid(x[:, :, :, ch:])


##################################################################################
# Normalization function
##################################################################################

def batch_norm(x, is_training=False, scope='batch_norm'):
    """
    if x_norm = tf.layers.batch_normalization
    # ...
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss)
    """

    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)

    # return tf.layers.batch_normalization(x, momentum=0.9, epsilon=1e-05, center=True, scale=True, training=is_training, name=scope)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


##################################################################################
# Loss function
##################################################################################

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y)) # [64, h, w, c]

    return loss

def discriminator_loss(gan_type, real_logit, fake_logit):
    real_loss = 0
    fake_loss = 0

    if real_logit is None :
        if gan_type == 'lsgan':
            fake_loss = tf.reduce_mean(tf.square(fake_logit))
        if gan_type == 'gan':
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logit), logits=fake_logit))

        if gan_type == 'hinge':
            fake_loss = tf.reduce_mean(relu(1 + fake_logit))
    else :
        if gan_type == 'lsgan':
            real_loss = tf.reduce_mean(tf.squared_difference(real_logit, 1.0))
            fake_loss = tf.reduce_mean(tf.square(fake_logit))

        if gan_type == 'gan':
            real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logit), logits=real_logit))
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logit), logits=fake_logit))

        if gan_type == 'hinge':

            real_loss = tf.reduce_mean(relu(1 - real_logit))
            fake_loss = tf.reduce_mean(relu(1 + fake_logit))

    return real_loss, fake_loss


def generator_loss(gan_type, fake_logit):
    fake_loss = 0

    if gan_type == 'lsgan':
        fake_loss = tf.reduce_mean(tf.squared_difference(fake_logit, 1.0))

    if gan_type == 'gan':
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logit), logits=fake_logit))

    if gan_type == 'hinge':
        fake_loss = -tf.reduce_mean(fake_logit)

    return fake_loss

def get_inception_feature(x) :
    from keras.applications.inception_v3 import preprocess_input as inception_preprocess
    from keras.applications.inception_v3 import InceptionV3
    from keras.models import Model

    x = resize(x, [299, 299])
    x = ((x + 1) / 2) * 255.0
    x = inception_preprocess(x)

    inception_v3_model = InceptionV3(weights='imagenet', include_top=False)
    inception_v3_model.trainable = False

    mixed_7_feature = Model(inputs=inception_v3_model.input, outputs=inception_v3_model.get_layer('mixed7').output)

    mixed_7_features = mixed_7_feature.predict(x)
    last_feature = inception_v3_model.predict(x)

    return mixed_7_features, last_feature

def regularization_loss(scope_name):
    """
    If you want to use "Regularization"
    g_loss += regularization_loss('generator')
    d_loss += regularization_loss('discriminator')
    """
    collection_regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss = []
    for item in collection_regularization:
        if scope_name in item.name:
            loss.append(item)

    return tf.reduce_sum(loss)


def kl_loss(mean, logvar):
    # shape : [batch_size, channel]
    # loss = 0.5 * tf.reduce_sum(tf.square(mean) + tf.exp(logvar) - 1 - logvar, axis=-1)
    # loss = tf.reduce_mean(loss)
    loss = 0.5 * tf.reduce_mean(tf.square(mean) + tf.exp(logvar) - 1 - logvar)

    return loss

##################################################################################
# Natural Language Processing
##################################################################################

def embed_sequence(x, n_words, embed_dim, init_range=0.1, trainable=True, scope='embed_layer') :
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) :
        embeddings = tf_contrib.layers.embed_sequence(x, n_words, embed_dim,
                                                      initializer=tf.random_uniform_initializer(minval=-init_range, maxval=init_range),
                                                      trainable=trainable)

        return embeddings
