import tensorflow as tf
import numpy as np

weight_initializer = tf.initializers.RandomNormal(mean=0.0, stddev=0.02)
weight_regularizer = tf.keras.regularizers.l2(0.0001)
weight_regularizer_fully = tf.keras.regularizers.l2(0.0001)


##################################################################################
# Layers
##################################################################################

# padding='SAME' ======> pad = floor[ (kernel - stride) / 2 ]
class Conv(tf.keras.layers.Layer):
    def __init__(self, channels, kernel=3, stride=1, pad=0, pad_type='zero', use_bias=True, sn=False, name='Conv'):
        super(Conv, self).__init__(name=name)
        self.channels = channels
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.pad_type = pad_type
        self.use_bias = use_bias
        self.sn = sn

        if self.sn :
            self.conv = SpectralNormalization(tf.keras.layers.Conv2D(filters=self.channels, kernel_size=self.kernel,
                                                                     kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer,
                                                                     strides=self.stride, use_bias=self.use_bias), name='sn_' + self.name)
        else :
            self.conv = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=self.kernel,
                                               kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer,
                                               strides=self.stride, use_bias=self.use_bias, name=self.name)

    def call(self, x, training=None, mask=None):
        if self.pad > 0:
            h = x.shape[1]
            if h % self.stride == 0:
                pad = self.pad * 2
            else:
                pad = max(self.kernel - (h % self.stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if self.pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')
            else:
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])

        x = self.conv(x)

        return x

class FullyConnected(tf.keras.layers.Layer):
    def __init__(self, units, use_bias=True, sn=False, name='FullyConnected'):
        super(FullyConnected, self).__init__(name=name)
        self.units = units
        self.use_bias = use_bias
        self.sn = sn

        if self.sn:
            self.fc = SpectralNormalization(tf.keras.layers.Dense(self.units,
                                                                  kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer_fully,
                                                                  use_bias=self.use_bias), name='sn_' + self.name)
        else :
            self.fc = tf.keras.layers.Dense(self.units,
                                            kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer_fully,
                                            use_bias=self.use_bias, name=self.name)

    def call(self, x, training=None, mask=None):
        x = flatten()(x)
        x = self.fc(x)

        return x

##################################################################################
# Blocks
##################################################################################

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, channels, name='ResBlock'):
        super(ResBlock, self).__init__(name=name)
        self.channels = channels

        self.conv_0 = Conv(self.channels * 2, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='conv_0')
        self.batch_norm_0 = BatchNorm(momentum=0.9, epsilon=1e-5, name='batch_norm_0')

        self.conv_1 = Conv(self.channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False,  name='conv_1')
        self.batch_norm_1 = BatchNorm(momentum=0.9, epsilon=1e-5, name='batch_norm_1')

    def call(self, x_init, training=None, mask=None):
        with tf.name_scope(self.name):
            with tf.name_scope('res1'):
                x = self.conv_0(x_init)
                x = self.batch_norm_0(x, training=training)
                x = GLU()(x)

            with tf.name_scope('res2'):
                x = self.conv_1(x)
                x = self.batch_norm_1(x, training=training)

            return x + x_init


##################################################################################
# Normalization
##################################################################################

class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, momentum=0.9, epsilon=1e-5, name='BatchNorm'):
        super(BatchNorm, self).__init__(name=name)
        self.momentum = momentum
        self.epsilon = epsilon

    def call(self, x, training=None, mask=None):
        x = tf.keras.layers.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon,
                                               center=True, scale=True,
                                               name=self.name)(x, training=training)
        return x

##################################################################################
# Activation Function
##################################################################################

def Leaky_Relu(x=None, alpha=0.01, name='leaky_relu'):
    # pytorch alpha is 0.01
    if x is None:
        return tf.keras.layers.LeakyReLU(alpha=alpha, name=name)
    else:
        return tf.keras.layers.LeakyReLU(alpha=alpha, name=name)(x)

def Relu(x=None, name='relu'):
    if x is None:
        return tf.keras.layers.Activation(tf.keras.activations.relu, name=name)

    else:
        return tf.keras.layers.Activation(tf.keras.activations.relu, name=name)(x)

class GLU(tf.keras.layers.Layer):
    def __init__(self):
        super(GLU, self).__init__()

    def build(self, input_shape):
        assert input_shape[-1] % 2 == 0, 'channels dont divide 2!'
        self.n_dim = len(input_shape)
        self.output_dim = input_shape[-1] // 2

    def call(self, x, training=None, mask=None):
        nc = self.output_dim
        if self.n_dim == 4:
            return x[:, :, :, :nc] * tf.sigmoid(x[:, :, :, nc:])
        if self.n_dim == 3:
            return x[:, :, :nc] * tf.sigmoid(x[:, :, nc:])
        if self.n_dim == 2:
            return x[:, :nc] * tf.sigmoid(x[:, nc:])

def Tanh(x=None, name='tanh'):
    if x is None:
        return tf.keras.layers.Activation(tf.keras.activations.tanh, name=name)
    else:
        return tf.keras.layers.Activation(tf.keras.activations.tanh, name=name)(x)

##################################################################################
# Pooling & Resize
##################################################################################

def resize(x, target_size):
    return tf.image.resize(x, size=target_size, method=tf.image.ResizeMethod.BILINEAR)

def nearest_up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize(x, size=new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

def flatten():
    return tf.keras.layers.Flatten()

##################################################################################
# Loss Function
##################################################################################

def regularization_loss(model):
    loss = tf.nn.scale_regularization_loss(model.losses)

    return loss

##################################################################################
# GAN Loss Function
##################################################################################
@tf.function
def discriminator_loss(gan_type, real_logit, fake_logit):
    real_loss = 0
    fake_loss = 0

    if real_logit is None :
        if gan_type == 'lsgan':
            fake_loss = tf.reduce_mean(tf.square(fake_logit))
        if gan_type == 'gan':
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logit), logits=fake_logit))

        if gan_type == 'hinge':
            fake_loss = tf.reduce_mean(Relu(1 + fake_logit))
    else :
        if gan_type == 'lsgan':
            real_loss = tf.reduce_mean(tf.math.squared_difference(real_logit, 1.0))
            fake_loss = tf.reduce_mean(tf.square(fake_logit))

        if gan_type == 'gan':
            real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logit), logits=real_logit))
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logit), logits=fake_logit))

        if gan_type == 'hinge':

            real_loss = tf.reduce_mean(Relu(1 - real_logit))
            fake_loss = tf.reduce_mean(Relu(1 + fake_logit))

    return real_loss, fake_loss

@tf.function
def generator_loss(gan_type, fake_logit):
    fake_loss = 0

    if gan_type == 'lsgan':
        fake_loss = tf.reduce_mean(tf.math.squared_difference(fake_logit, 1.0))

    if gan_type == 'gan':
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logit), logits=fake_logit))

    if gan_type == 'hinge':
        fake_loss = -tf.reduce_mean(fake_logit)

    return fake_loss

@tf.function
def L2_loss(x, y):
    loss = tf.reduce_mean(tf.square(x - y))

    return loss

@tf.function
def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss

##################################################################################
# KL-Divergence Loss Function
##################################################################################

def reparametrize(mean, logvar):
    eps = tf.random.normal(tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32)

    return mean + tf.exp(logvar * 0.5) * eps

@tf.function
def kl_loss(mean, logvar):
    # shape : [batch_size, channel]
    loss = 0.5 * tf.reduce_sum(tf.square(mean) + tf.exp(logvar) - 1 - logvar, axis=-1)
    loss = tf.reduce_mean(loss)

    return loss

##################################################################################
# Class function
##################################################################################

class get_weight(tf.keras.layers.Layer):
    def __init__(self, w_shape, w_init, w_regular, w_trainable):
        super(get_weight, self).__init__()

        self.w_shape = w_shape
        self.w_init = w_init
        self.w_regular = w_regular
        self.w_trainable = w_trainable
        # self.w_name = w_name

    def call(self, inputs=None, training=None, mask=None):
        return self.add_weight(shape=self.w_shape, dtype=tf.float32,
                               initializer=self.w_init, regularizer=self.w_regular,
                               trainable=self.w_trainable)


class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape=None):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name=self.name + '_u',
                                 dtype=tf.float32, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        super(SpectralNormalization, self).build()

    def call(self, inputs, training=None, mask=None):
        self.update_weights()
        output = self.layer(inputs)
        # self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = None

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)

        self.layer.kernel = self.w / sigma

    def restore_weights(self):

        self.layer.kernel = self.w

##################################################################################
# Natural Language Processing
##################################################################################

class VariousRNN(tf.keras.layers.Layer):
    def __init__(self, n_hidden=128, n_layer=1, dropout_rate=0.5, bidirectional=True, return_state=True, rnn_type='lstm', name='VariousRNN'):
        super(VariousRNN, self).__init__(name=name)
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.return_state = return_state
        self.rnn_type = rnn_type.lower()

        if self.rnn_type == 'lstm':
            self.cell_type = tf.keras.layers.LSTMCell
        elif self.rnn_type == 'gru':
            self.cell_type = tf.keras.layers.GRUCell
        else:
            raise NotImplementedError

        self.rnn = tf.keras.layers.RNN([self.cell_type(units=n_hidden, dropout=self.dropout_rate) for _ in range(self.n_layer)], return_sequences=True, return_state=self.return_state)
        if self.bidirectional:
            self.rnn = tf.keras.layers.Bidirectional(self.rnn)
        """
        if also return_state=True, 
        whole_sequence, forward_hidden, forward_cell, backward_hidden, backward_cell (LSTM)
        whole_sequence, forward_hidden, forward_cell (GRU)
        sent_emb = tf.concat([forward_hidden, backward_hidden], axis=-1)
        """

    def call(self, x, training=None, mask=None):
        if self.return_state:
            if self.bidirectional:
                if self.rnn_type == 'gru':
                    output, forward_h, backward_h = self.rnn(x, training=training)
                else : # LSTM
                    output, forward_state, backward_state = self.rnn(x, training=training)
                    forward_h, backward_h = forward_state[0], backward_state[0]
                    forward_c, backward_c = forward_state[1], backward_state[1]

                sent_emb = tf.concat([forward_h, backward_h], axis=-1)
            else :
                if self.rnn_type =='gru':
                    output, forward_h = self.rnn(x, training=training)
                else :
                    output, forward_state = self.rnn(x, training=training)
                    forward_h, forward_c = forward_state

                sent_emb = forward_h

        else :
            output = self.rnn(x, training=training)
            sent_emb = output[:, -1, :]

        word_emb = output

        return word_emb, sent_emb

def EmbedSequence(n_words, embed_dim, trainable=True, name='embed_layer') :

    emeddings = tf.keras.layers.Embedding(input_dim=n_words, output_dim=embed_dim,
                                          trainable=trainable, name=name)
    return emeddings

class DropOut(tf.keras.layers.Layer):
    def __init__(self, drop_rate=0.5, name='DropOut'):
        super(DropOut, self).__init__(name=name)
        self.drop_rate = drop_rate

    def call(self, x, training=None, mask=None):
        x = tf.keras.layers.Dropout(self.drop_rate, name=self.name)(x, training=training)
        return x

def caption_loss(cap_output, captions):
    # log-softmax cross_entropy loss
    # https://stevensmit.me/softmax-or-log-softmax-for-cross-entropy-loss-in-tensorflow/


    loss = tf.nn.softmax_cross_entropy_with_logits(logits=cap_output, labels=captions)
    loss = tf.reduce_mean(loss)

    # cap_output = tf.nn.log_softmax(cap_output)
    # loss = tf.reduce_sum(-1 * tf.math.multiply(captions, cap_output), axis=-1)
    # loss = tf.reduce_mean(loss)

    # loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=cap_output, labels=captions)
    # loss = tf.reduce_mean(loss)

    return loss

def get_accuracy(logit, label):
    prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return accuracy

def word_level_correlation_loss(img_feature, word_emb, gamma1=4.0, gamma2=5.0):

    # img_feature = [bs, 17, 17, 256] = context
    # word_emb = [bs, seq_len, 256 = hidden * 2] = query

    # func_attention
    batch_size = img_feature.shape[0]
    seq_len = word_emb.shape[1]
    similar_list = []

    for i in range(batch_size) :
        context = tf.expand_dims(img_feature[i], axis=0)
        word = tf.expand_dims(word_emb[i], axis=0)

        weighted_context, attn = func_attention(context, word, gamma1)
        # weighted_context = [bs, 256, seq_len]
        # attn = [bs, h, w, seq_len]

        aver_word = tf.reduce_mean(word, axis=1, keepdims=True) # [bs, 1, 256]

        res_word = tf.matmul(aver_word, word, transpose_b=True) # [bs, 1, seq_len]
        res_word_softmax = tf.nn.softmax(res_word, axis=1)
        res_word_softmax = tf.tile(res_word_softmax, multiples=[1, weighted_context.shape[1], 1]) # [bs, 256, seq_len]

        self_weighted_context = tf.transpose(weighted_context * res_word_softmax, perm=[0, 2, 1]) # [bs, seq_len, 256]

        word = tf.reshape(word, [seq_len, -1]) # [seq_len, 256]
        self_weighted_context = tf.reshape(self_weighted_context, [seq_len, -1]) # [seq_len, 256]

        row_sim = cosine_similarity(word, self_weighted_context) #[seq_len]

        row_sim = tf.exp(row_sim * gamma2)
        row_sim = tf.reduce_sum(row_sim) # []
        row_sim = tf.math.log(row_sim)

        similar_list.append(row_sim)

    word_match_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=similar_list, labels=tf.ones_like(similar_list)))
    word_mismatch_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=similar_list, labels=tf.zeros_like(similar_list)))

    loss = (word_match_loss + word_mismatch_loss) / 2.0

    return loss

def func_attention(img_feature, word_emb, gamma1=4.0):
    # word_emb = query
    # img_feature = context
    # 256 = self.emb_dim

    bs, seq_len = word_emb.shape[0], word_emb.shape[1] # seq_len = length of query
    h, w = img_feature.shape[1], img_feature.shape[2]
    hw = h * w # length of source

    # context = [bs, 17, 17, 256]
    # query = [bs, seq_len, 256]
    # 256 = ndf
    context = tf.reshape(img_feature, [bs, hw, -1]) # [bs, hw, 256]
    attn = tf.matmul(context, word_emb, transpose_b=True) # [bs, hw, seq_len]
    attn = tf.reshape(attn, [bs*hw, seq_len])
    attn = tf.nn.softmax(attn)

    attn = tf.reshape(attn, [bs, hw, seq_len])
    attn = tf.transpose(attn, perm=[0, 2, 1])
    attn = tf.reshape(attn, [bs*seq_len, hw])

    attn = attn * gamma1
    attn = tf.nn.softmax(attn)
    attn = tf.reshape(attn, [bs, seq_len, hw])

    weighted_context = tf.matmul(context, attn, transpose_a=True, transpose_b=True) # [bs, 256, seq_len]

    return weighted_context, tf.reshape(tf.transpose(attn, [0, 2, 1]), [bs, h, w, seq_len])

def cosine_similarity(x, y):

    xy = tf.reduce_sum(x * y, axis=-1)
    x = tf.norm(x, axis=-1)
    y = tf.norm(y, axis=-1)

    similarity = (xy / ((x * y) + 1e-8))

    return similarity

def normalization(x):
    x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))
    return x

def word_loss(img_feature, word_emb, class_id, gamma2=5.0):
    batch_size = word_emb.shape[0]
    seq_len = word_emb.shape[1]

    label = tf.cast(range(batch_size), tf.int32)
    masks = []
    similarities = []

    for i in range(batch_size):
        mask = (class_id.numpy() == class_id[i].numpy()).astype(np.uint8)
        mask[i] = 0
        masks.append(np.reshape(mask, newshape=[1, -1]))

        word = tf.expand_dims(word_emb[i, :, :], axis=0) # [1, seq_len, embed_dim]
        word = tf.tile(word, multiples=[batch_size, 1, 1])

        context = img_feature

        weiContext, _ = func_attention(context, word)
        weiContext = tf.transpose(weiContext, perm=[0, 2, 1]) # [bs, seq_len, embed_dim=256]

        word = tf.reshape(word, shape=[batch_size * seq_len, -1])
        weiContext = tf.reshape(weiContext, shape=[batch_size * seq_len, -1])

        row_sim = cosine_similarity(word, weiContext)
        row_sim = tf.reshape(row_sim, shape=[batch_size, seq_len])

        row_sim = tf.exp(row_sim * gamma2)
        row_sim = tf.reduce_sum(row_sim, axis=-1, keepdims=True)
        row_sim = tf.math.log(row_sim)

        similarities.append(row_sim)

    similarities = tf.concat(similarities, axis=-1)
    masks = tf.cast(tf.concat(masks, axis=0), tf.float32)

    similarities = similarities * gamma2

    similarities = tf.where(tf.equal(masks, True), x=tf.constant(-float('inf'), dtype=tf.float32, shape=masks.shape), y=similarities)

    similarities1 = tf.transpose(similarities, perm=[1, 0])

    loss0 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=similarities, labels=label))
    loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=similarities1, labels=label))

    loss = loss0 + loss1

    return loss

def sent_loss(img_feature, sent_emb, class_id, gamma3=10.0):
    batch_size = sent_emb.shape[0]
    label = tf.cast(range(batch_size), tf.int32)

    masks = []

    for i in range(batch_size):
        mask = (class_id.numpy() == class_id[i].numpy()).astype(np.uint8)
        mask[i] = 0
        masks.append(np.reshape(mask, newshape=[1, -1]))

    masks = tf.cast(tf.concat(masks, axis=0), tf.float32)

    cnn_code = tf.expand_dims(img_feature, axis=0)
    rnn_code = tf.expand_dims(sent_emb, axis=0)

    cnn_code_norm = tf.norm(cnn_code, axis=-1, keepdims=True)
    rnn_code_norm = tf.norm(rnn_code, axis=-1, keepdims=True)

    scores0 = tf.matmul(cnn_code, rnn_code, transpose_b=True)
    norm0 = tf.matmul(cnn_code_norm, rnn_code_norm, transpose_b=True)
    scores0 = scores0 / tf.clip_by_value(norm0, clip_value_min=1e-8, clip_value_max=float('inf')) * gamma3

    scores0 = tf.squeeze(scores0, axis=0)

    scores0 = tf.where(tf.equal(masks, True), x=tf.constant(-float('inf'), dtype=tf.float32, shape=masks.shape), y=scores0)
    scores1 = tf.transpose(scores0, perm=[1, 0])

    loss0 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores0, labels=label))
    loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores1, labels=label))

    loss = loss0 + loss1

    return loss