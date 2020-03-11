from ops import *
from tensorflow.keras import Sequential


##################################################################################
# Generator
##################################################################################
class CnnEncoder(tf.keras.Model):
    def __init__(self, embed_dim, name='CnnEncoder'):
        super(CnnEncoder, self).__init__(name=name)
        self.embed_dim = embed_dim

        self.inception_v3_preprocess = tf.keras.applications.inception_v3.preprocess_input
        self.inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        self.inception_v3.trainable = False

        self.inception_v3_mixed7 = tf.keras.Model(inputs=self.inception_v3.input, outputs=self.inception_v3.get_layer('mixed7').output)
        self.inception_v3_mixed7.trainable = False

        self.emb_feature = Conv(channels=self.embed_dim, kernel=1, stride=1, use_bias=False, name='emb_feature_conv') # word_feature
        self.emb_code = FullyConnected(units=self.embed_dim, use_bias=True, name='emb_code_fc') # sent code

    def call(self, x, training=True, mask=None):
        x = ((x + 1) / 2) * 255.0
        x = resize(x, [299, 299])
        x = self.inception_v3_preprocess(x)

        code = self.inception_v3(x)
        feature = self.inception_v3_mixed7(x)

        feature = self.emb_feature(feature)
        code = self.emb_code(code)

        return feature, code

class RnnEncoder(tf.keras.Model):
    def __init__(self, n_words, embed_dim=256, drop_rate=0.5, n_hidden=128, n_layer=1, bidirectional=True, rnn_type='lstm', name='RnnEncoder'):
        super(RnnEncoder, self).__init__(name=name)
        self.n_words = n_words
        self.embed_dim = embed_dim
        self.drop_rate = drop_rate
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type

        self.model = self.architecture()
        self.rnn = VariousRNN(self.n_hidden, self.n_layer, self.drop_rate, self.bidirectional, rnn_type=self.rnn_type, name=self.rnn_type + '_rnn')

    def architecture(self):
        model = []

        model += [EmbedSequence(self.n_words, self.embed_dim, name='embed_layer')] # [bs, seq_len, embed_dim]
        model += [DropOut(self.drop_rate, name='dropout')]

        model = Sequential(model)

        return model


    def call(self, caption, training=True, mask=None):
        # caption = [bs, seq_len]
        x = self.model(caption, training=training)
        word_emb, sent_emb = self.rnn(x, training=training)  # (bs, seq_len, n_hidden * 2) (bs, n_hidden * 2)
        mask = tf.equal(caption, 0)

        # 일단은 mask return 안함 (pytorch)
        # n_hidden * 2 = embed_dim

        return word_emb, sent_emb, mask

class CA_NET(tf.keras.Model):
    def __init__(self, c_dim, name='CA_NET'):
        super(CA_NET, self).__init__(name=name)
        self.c_dim = c_dim # z_dim, condition dimension

        self.model = self.architecture()

    def architecture(self):
        model = []

        model += [FullyConnected(units=self.c_dim * 2, name='mu_fc')]
        model += [Relu()]

        model = Sequential(model)

        return model

    def call(self, sent_emb, training=True, mask=None):
        x = self.model(sent_emb, training=training)

        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]

        c_code = reparametrize(mu, logvar)

        return c_code, mu, logvar

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, channels, name='SpatialAttention'):
        super(SpatialAttention, self).__init__(name=name)
        self.channels = channels # idf, x.shape[-1]

        self.word_conv = Conv(self.channels, kernel=1, stride=1, use_bias=False, name='word_conv')
        self.sentence_fc = FullyConnected(units=self.channels, name='sent_fc')
        self.sentence_conv = Conv(self.channels, kernel=1, stride=1, use_bias=False, name='sentence_conv')

    def build(self, input_shape):
        self.bs, self.h, self.w, _ = input_shape[0]
        self.hw = self.h * self.w # length of query
        self.seq_len = input_shape[2][1] # length of source

    def call(self, inputs, training=True):
        x, sentence, context, mask = inputs # context = word_emb
        x = tf.reshape(x, shape=[self.bs, self.hw, -1])

        context = tf.expand_dims(context, axis=1)
        context = self.word_conv(context)
        context = tf.squeeze(context, axis=1)

        attn = tf.matmul(x, context, transpose_b=True) # [bs, hw, seq_len]
        attn = tf.reshape(attn, shape=[self.bs * self.hw, self.seq_len])

        mask = tf.tile(mask, multiples=[self.hw, 1])
        attn = tf.where(tf.equal(mask, True), x=tf.constant(-float('inf'), dtype=tf.float32, shape=mask.shape), y=attn)
        attn = tf.nn.softmax(attn)
        attn = tf.reshape(attn, shape=[self.bs, self.hw, self.seq_len])

        weighted_context = tf.matmul(context, attn, transpose_a=True, transpose_b=True)
        weighted_context = tf.reshape(tf.transpose(weighted_context, perm=[0, 2, 1]), shape=[self.bs, self.h, self.w, -1])
        word_attn = tf.reshape(attn, shape=[self.bs, self.h, self.w, -1])

        return weighted_context, word_attn


class UpBlock(tf.keras.layers.Layer):
    def __init__(self, channels, name='UpBlock'):
        super(UpBlock, self).__init__(name=name)
        self.channels = channels

        self.model = self.architecture()

    def architecture(self):
        model = []

        model += [Conv(self.channels * 2, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='conv')]
        model += [BatchNorm(name='batch_norm')]
        model += [GLU()]

        model = Sequential(model)

        return model

    def call(self, x_init, training=True):
        x = nearest_up_sample(x_init, scale_factor=2)

        x = self.model(x, training=training)

        return x

class Generator_64(tf.keras.layers.Layer):
    def __init__(self, channels, name='Generator_64'):
        super(Generator_64, self).__init__(name=name)
        self.channels = channels

        self.model, self.generate_img_block = self.architecture()

    def architecture(self):
        model = []

        model += [FullyConnected(units=self.channels * 4 * 4 * 2, use_bias=False, name='code_fc')]
        model += [BatchNorm(name='batch_norm')]
        model += [GLU()]
        model += [tf.keras.layers.Reshape(target_shape=[4, 4, self.channels])]

        for i in range(4):
            model += [UpBlock(self.channels // 2, name='up_block_' + str(i))]
            self.channels = self.channels // 2

        model = Sequential(model)

        generate_img_block = []
        generate_img_block += [Conv(channels=3, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='g_64_logit')]
        generate_img_block += [Tanh()]

        generate_img_block = Sequential(generate_img_block)

        return model, generate_img_block


    def call(self, c_z_code, training=True, mask=None):
        h_code = self.model(c_z_code, training=training)
        x = self.generate_img_block(h_code, training=training)

        return h_code, x


class Generator_128(tf.keras.layers.Layer):
    def __init__(self, channels, name='Generator_128'):
        super(Generator_128, self).__init__(name=name)
        self.channels = channels # gf_dim

        self.spatial_attention = SpatialAttention(channels=self.channels)

        self.model, self.generate_img_block = self.architecture()

    def architecture(self):
        model = []

        for i in range(2):
            model += [ResBlock(self.channels * 2, name='resblock_' + str(i))]

        model += [UpBlock(self.channels, name='up_block')]

        model = Sequential(model)

        generate_img_block = []
        generate_img_block += [Conv(channels=3, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='g_128_logit')]
        generate_img_block += [Tanh()]

        generate_img_block = Sequential(generate_img_block)

        return model, generate_img_block

    def call(self, inputs, training=True):
        h_code, c_code, word_emb, mask = inputs
        c_code, _ = self.spatial_attention([h_code, c_code, word_emb, mask])

        h_c_code = tf.concat([h_code, c_code], axis=-1)

        h_code = self.model(h_c_code, training=training)
        x = self.generate_img_block(h_code)

        return c_code, h_code, x

class Generator_256(tf.keras.layers.Layer):
    def __init__(self, channels, name='Generator_256'):
        super(Generator_256, self).__init__(name=name)
        self.channels = channels

        self.spatial_attention = SpatialAttention(channels=self.channels)
        self.model = self.architecture()

    def architecture(self):
        model = []

        for i in range(2):
            model += [ResBlock(self.channels * 2, name='res_block_' + str(i))]

        model += [UpBlock(self.channels, name='up_block')]

        model += [Conv(channels=3, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='g_256_logit')]
        model += [Tanh()]

        model = Sequential(model)

        return model

    def call(self, inputs, training=True):
        h_code, c_code, word_emb, mask = inputs
        c_code, _ = self.spatial_attention([h_code, c_code, word_emb, mask])

        h_c_code = tf.concat([h_code, c_code], axis=-1)

        x = self.model(h_c_code, training=training)

        return x

class Generator(tf.keras.Model):
    def __init__(self, channels, name='Generator'):
        super(Generator, self).__init__(name=name)
        self.channels = channels

        # self.c_dim = c_dim
        # self.ca_net = CA_NET(self.c_dim)

        self.g_64 = Generator_64(self.channels * 16, name='g_64')
        self.g_128 = Generator_128(self.channels, name='g_128')
        self.g_256 = Generator_256(self.channels, name='g_256')

    def call(self, inputs, training=True, mask=None):

        # z_code, sent_emb, word_emb, mask = inputs
        # c_code, mu, logvar = self.ca_net(sent_emb, training=training)

        c_code, z_code, word_emb, mask = inputs
        c_z_code = tf.concat([c_code, z_code], axis=-1)

        h_code1, x_64 = self.g_64(c_z_code, training=training)
        c_code, h_code2, x_128 = self.g_128([h_code1, c_code, word_emb, mask], training=training)
        x_256 = self.g_256([h_code2, c_code, word_emb, mask], training=training)

        x = [x_64, x_128, x_256]

        return x


##################################################################################
# Discriminator
##################################################################################

class DownBlock(tf.keras.layers.Layer):
    def __init__(self, channels, name='DownBlock'):
        super(DownBlock, self).__init__(name=name)
        self.channels = channels

        self.model = self.architecture()

    def architecture(self):
        model = []

        model += [Conv(self.channels, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, name='conv')]
        model += [BatchNorm(name='batch_norm')]
        model += [Leaky_Relu(alpha=0.2)]

        model = Sequential(model)

        return model

    def call(self, x, training=True):
        x = self.model(x, training=training)

        return x

class Discriminator_64(tf.keras.layers.Layer):
    def __init__(self, channels, name='Discriminator_64'):
        super(Discriminator_64, self).__init__(name=name)
        self.channels = channels # self.df_dim

        self.uncond_logit_conv = Conv(channels=1, kernel=4, stride=4, use_bias=True, name='uncond_d_logit')
        self.cond_logit_conv = Conv(channels=1, kernel=4, stride=4, use_bias=True, name='cond_d_logit')
        self.model, self.code_block = self.architecture()

    def architecture(self):
        model = []

        model += [Conv(self.channels, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, name='convv')]
        model += [Leaky_Relu(alpha=0.2)]

        for i in range(3):
            model += [Conv(self.channels * 2, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, name='conv_' + str(i))]
            model += [BatchNorm(name='batch_norm_' + str(i))]
            model += [Leaky_Relu(alpha=0.2)]

            self.channels = self.channels * 2

        model = Sequential(model)

        code_block = []
        code_block += [Conv(self.channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='conv_code')]
        code_block += [BatchNorm(name='batch_norm_code')]
        code_block += [Leaky_Relu(alpha=0.2)]

        code_block = Sequential(code_block)

        return model, code_block

    def call(self, inputs, training=True):
        x, sent_emb = inputs

        x = self.model(x, training=training)

        # uncondition
        uncond_logit = self.uncond_logit_conv(x)

        # condition
        h_c_code = tf.concat([x, sent_emb], axis=-1)
        h_c_code = self.code_block(h_c_code, training=training)

        cond_logit = self.cond_logit_conv(h_c_code)

        return uncond_logit, cond_logit

class Discriminator_128(tf.keras.layers.Layer):
    def __init__(self, channels, name='Discriminator_128'):
        super(Discriminator_128, self).__init__(name=name)
        self.channels = channels

        self.uncond_logit_conv = Conv(channels=1, kernel=4, stride=4, use_bias=True, name='uncond_d_logit')
        self.cond_logit_conv = Conv(channels=1, kernel=4, stride=4, use_bias=True, name='cond_d_logit')
        self.model, self.code_block = self.architecture()


    def architecture(self):
        model = []

        model += [Conv(self.channels, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, name='conv')]
        model += [Leaky_Relu(alpha=0.2)]

        for i in range(3):
            model += [Conv(self.channels * 2, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, name='conv_' + str(i))]
            model += [BatchNorm(name='batch_norm_' + str(i))]
            model += [Leaky_Relu(alpha=0.2)]

            self.channels = self.channels * 2

        model += [DownBlock(self.channels * 2, name='down_block')]

        model += [Conv(self.channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='last_conv')]
        model += [BatchNorm(name='last_batch_norm')]
        model += [Leaky_Relu(alpha=0.2)]

        model = Sequential(model)

        code_block = []
        code_block += [Conv(self.channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='conv_code')]
        code_block += [BatchNorm(name='batch_norm_code')]
        code_block += [Leaky_Relu(alpha=0.2)]

        code_block = Sequential(code_block)

        return model, code_block


    def call(self, inputs, training=True):
        x, sent_emb = inputs

        x = self.model(x, training=training)

        # uncondition
        uncond_logit = self.uncond_logit_conv(x)

        # condition
        h_c_code = tf.concat([x, sent_emb], axis=-1)
        h_c_code = self.code_block(h_c_code, training=training)

        cond_logit = self.cond_logit_conv(h_c_code)

        return uncond_logit, cond_logit

class Discriminator_256(tf.keras.layers.Layer):
    def __init__(self, channels, name='Discriminator_256'):
        super(Discriminator_256, self).__init__(name=name)
        self.channels = channels

        self.uncond_logit_conv = Conv(channels=1, kernel=4, stride=4, use_bias=True, name='uncond_d_logit')
        self.cond_logit_conv = Conv(channels=1, kernel=4, stride=4, use_bias=True, name='cond_d_logit')
        self.model, self.code_block = self.architecture()


    def architecture(self):
        model = []

        model += [Conv(self.channels, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, name='conv')]
        model += [Leaky_Relu(alpha=0.2)]

        for i in range(3):
            model += [Conv(self.channels * 2, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, name='conv_' + str(i))]
            model += [BatchNorm(name='batch_norm_' + str(i))]
            model += [Leaky_Relu(alpha=0.2)]

            self.channels = self.channels * 2

        for i in range(2):
            model += [DownBlock(self.channels * 2, name='down_block_' + str(i))]

        for i in range(2):
            model += [Conv(self.channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='last_conv_' + str(i))]
            model += [BatchNorm(name='last_batch_norm_' + str(i))]
            model += [Leaky_Relu(alpha=0.2)]

        model = Sequential(model)

        code_block = []
        code_block += [Conv(self.channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='conv_code')]
        code_block += [BatchNorm(name='batch_norm_code')]
        code_block += [Leaky_Relu(alpha=0.2)]

        code_block = Sequential(code_block)

        return model, code_block


    def call(self, inputs, training=True):
        x, sent_emb = inputs

        x = self.model(x, training=training)

        # uncondition
        uncond_logit = self.uncond_logit_conv(x)

        # condition
        h_c_code = tf.concat([x, sent_emb], axis=-1)
        h_c_code = self.code_block(h_c_code, training=training)

        cond_logit = self.cond_logit_conv(h_c_code)

        return uncond_logit, cond_logit

class Discriminator(tf.keras.Model):
    def __init__(self, channels, embed_dim, name='Discriminator'):
        super(Discriminator, self).__init__(name=name)
        self.channels = channels
        self.embed_dim = embed_dim

        self.d_64 = Discriminator_64(self.channels, name='d_64')
        self.d_128 = Discriminator_128(self.channels, name='d_128')
        self.d_256 = Discriminator_256(self.channels, name='d_256')

    def call(self, inputs, training=True, mask=None):
        x_64, x_128, x_256, sent_emb = inputs
        sent_emb = tf.reshape(sent_emb, shape=[-1, 1, 1, self.embed_dim])
        sent_emb = tf.tile(sent_emb, multiples=[1, 4, 4, 1])

        x_64_uncond_logit, x_64_cond_logit = self.d_64([x_64, sent_emb], training=training)
        x_128_uncond_logit, x_128_cond_logit = self.d_128([x_128, sent_emb], training=training)
        x_256_uncond_logit, x_256_cond_logit = self.d_256([x_256, sent_emb], training=training)

        uncond_logits = [x_64_uncond_logit, x_128_uncond_logit, x_256_uncond_logit]
        cond_logits = [x_64_cond_logit, x_128_cond_logit, x_256_cond_logit]

        return uncond_logits, cond_logits