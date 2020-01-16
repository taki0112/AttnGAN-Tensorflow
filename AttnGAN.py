from ops import *
from utils import *
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import numpy as np

class AttnGAN():
    def __init__(self, sess, args):

        self.phase = args.phase
        self.model_name = 'AttnGAN'

        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.decay_iter = args.decay_iter

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.init_lr = args.lr

        self.gan_type = args.gan_type

        self.df_dim = args.df_dim
        self.gf_dim = args.gf_dim
        self.embed_dim = args.embed_dim
        self.z_dim = args.z_dim


        """ Weight """
        self.adv_weight = args.adv_weight
        self.kl_weight = args.kl_weight


        """ Generator """

        """ Discriminator """
        self.sn = args.sn

        self.img_height = args.img_height
        self.img_width = args.img_width

        self.img_ch = args.img_ch

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.dataset_path = os.path.join('./dataset', self.dataset_name)

        print()

        print("##### Information #####")
        print("# dataset : ", self.dataset_name)
        print("# batch_size : ", self.batch_size)
        print("# max iteration : ", self.iteration)
        print("# z_dim : ", self.z_dim)
        print("# embed_dim : ", self.embed_dim)

        print()

        print("##### Generator #####")
        print("# ")
        print()

        print("##### Discriminator #####")
        print("# spectral normalization : ", self.sn)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# kl_weight : ", self.kl_weight)

        print()

    ##################################################################################
    # Generator
    ##################################################################################

    def rnn_encoder(self, caption, n_words, is_training=True,
                    embed_dim=300, drop_rate=0.5, n_hidden=128, n_layers=1, bidirectional=True, rnn_type='lstm',
                    reuse=tf.AUTO_REUSE, scope='rnn_encoder'):
        with tf.variable_scope(scope, reuse=reuse):
            # caption = [bs, seq_len]
            embeddings = embed_sequence(caption, n_words=n_words, embed_dim=embed_dim, trainable=True, scope='embed_layer')
            embeddings = dropout(embeddings, drop_rate, is_training)

            words_emb, sent_emb = various_rnn(embeddings, n_layers, n_hidden, drop_rate, bidirectional, rnn_type=rnn_type, scope='rnn')
            mask = tf.equal(caption, 0)

            # n_hidden * 2 = embed_dim
            # (bs, seq_len, n_hidden * 2) (bs, n_hidden * 2) (bs, seq_len)

            return words_emb, sent_emb, mask

    def ca_net(self, text_emb, reuse=tf.AUTO_REUSE, scope='ca_net'):
        with tf.variable_scope(scope, reuse=reuse):
            mu = fully_connected(text_emb, units=self.z_dim, use_bias=True, sn=self.sn, scope='mu_fc')
            mu = relu(mu)

            logvar = fully_connected(text_emb, units=self.z_dim, use_bias=True, sn=self.sn, scope='logvar_fc')
            logvar = relu(logvar)

            c_code = reparametrize(mu, logvar)

            return c_code, mu, logvar

    def generator(self, z_code, sent_emb, word_emb, mask, is_training=True, reuse=tf.AUTO_REUSE, scope='generator'):
        channels = self.gf_dim * 16 # 32 * 4 * 4
        with tf.variable_scope(scope, reuse=reuse):
            fake_imgs = []
            att_maps = []
            c_code, mu, logvar = self.ca_net(sent_emb)

            # 64 img
            c_z_code = tf.concat([c_code, z_code], axis=-1)
            out_code = fully_connected(c_z_code, units=channels * 4 * 4 * 2, use_bias=False, sn=self.sn, scope='code_fc')
            out_code = batch_norm(out_code, is_training)
            out_code = glu(out_code)
            h_code1 = tf.reshape(out_code, shape=[-1, 4, 4, channels])

            for i in range(4) :
                h_code1 = up_block(h_code1, channels // 2, is_training, use_bias=False, sn=self.sn, scope='64_up_block_' + str(i))
                channels = channels // 2

            x_64 = conv(h_code1, channels=3, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='64_conv')
            x_64 = tanh(x_64)

            fake_imgs.append(x_64)
            att_maps.append(None)

            # 128 img
            channels = self.gf_dim
            c_code, att_1 = attention_net(h_code1, word_emb, mask, channels, use_bias=False, sn=self.sn, scope='128_attention_net')
            h_c_code = tf.concat([h_code1, c_code], axis=-1)

            for i in range(2):
                h_c_code = resblock(h_c_code, channels * 2, is_training, use_bias=False, sn=self.sn, scope='128_resblock_' + str(i))

            h_code2 = up_block(h_c_code, channels, is_training, use_bias=False, sn=self.sn, scope='128_up_block_' + str(i))

            x_128 = conv(h_code2, channels=3, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='128_conv')
            x_128 = tanh(x_128)

            fake_imgs.append(x_128)
            att_maps.append(att_1)

            # 256 img
            c_code, att_2 = attention_net(h_code2, word_emb, mask, channels, use_bias=False, sn=self.sn, scope='256_attention_net')
            h_c_code = tf.concat([h_code2, c_code], axis=-1)

            for i in range(2):
                h_c_code = resblock(h_c_code, channels * 2, is_training, use_bias=False, sn=self.sn, scope='256_resblock_' + str(i))

            h_code3 = up_block(h_c_code, channels, is_training, use_bias=False, sn=self.sn, scope='256_up_block_' + str(i))

            x_256 = conv(h_code3, channels=3, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='256_conv')
            x_256 = tanh(x_256)

            fake_imgs.append(x_256)
            att_maps.append(att_2)

            return fake_imgs, att_maps, mu, logvar


    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, sent_emb, is_training=True, reuse=tf.AUTO_REUSE, scope='discriminator'):
        sent_emb = tf.reshape(sent_emb, shape=[-1, 1, 1, self.embed_dim])
        sent_emb = tf.tile(sent_emb, multiples=[1, 4, 4, 1])

        uncond_logits = []
        cond_logits = []

        x_64, x_128, x_256 = x_init[0], x_init[1], x_init[2]

        with tf.variable_scope(scope+'_64', reuse=reuse):
            channel = self.df_dim
            x = conv(x_64, channel, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='conv')
            x = lrelu(x, 0.2)

            for i in range(3):
                x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='conv_' + str(i))
                x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                x = lrelu(x, 0.2)

                channel = channel * 2

            # uncondition
            uncond_logit = conv(x, channels=1, kernel=4, stride=4, use_bias=True, sn=self.sn, scope='uncond_logit')

            # condition
            h_c_code = tf.concat([x, sent_emb], axis=-1)
            h_c_code = conv(h_c_code, channel, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='conv_code')
            h_c_code = batch_norm(h_c_code, is_training, scope='batch_norm_code')
            h_c_code = lrelu(h_c_code, 0.2)

            cond_logit = conv(h_c_code, channels=1, kernel=4, stride=4, use_bias=True, sn=self.sn, scope='cond_logit')

            uncond_logits.append(uncond_logit)
            cond_logits.append(cond_logit)

        with tf.variable_scope(scope+'_128', reuse=reuse):
            channel = self.df_dim
            x = conv(x_128, channel, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='conv')
            x = lrelu(x, 0.2)

            for i in range(3):
                x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='conv_' + str(i))
                x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                x = lrelu(x, 0.2)

                channel = channel * 2

            x = down_block(x, channel * 2, is_training, use_bias=False, sn=self.sn, scope='down_block_0')

            x = conv(x, channel, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='conv_3x3_0')
            x = batch_norm(x, is_training, scope='batch_norm_code')
            x = lrelu(x, 0.2)

            # uncondition
            uncond_logit = conv(x, channels=1, kernel=4, stride=4, use_bias=True, sn=self.sn, scope='uncond_logit')

            # condition
            h_c_code = tf.concat([x, sent_emb], axis=-1)
            h_c_code = conv(h_c_code, channel, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='conv_code')
            h_c_code = batch_norm(h_c_code, is_training, scope='batch_norm_code')
            h_c_code = lrelu(h_c_code, 0.2)

            cond_logit = conv(h_c_code, channels=1, kernel=4, stride=4, use_bias=True, sn=self.sn, scope='cond_logit')

            uncond_logits.append(uncond_logit)
            cond_logits.append(cond_logit)

        with tf.variable_scope(scope+'_256', reuse=reuse):
            channel = self.df_dim
            x = conv(x_256, channel, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='conv')
            x = lrelu(x, 0.2)

            for i in range(3):
                x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='conv_' + str(i))
                x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                x = lrelu(x, 0.2)

                channel = channel * 2

            for i in range(2):
                x = down_block(x, channel * 2, is_training, use_bias=False, sn=self.sn, scope='down_block_' + str(i))

                channel = channel * 2

            for i in range(2):
                x = conv(x, channel // 2, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='conv_3x3_' + str(i))
                x = batch_norm(x, is_training, scope='batch_norm_3x3_' + str(i))
                x = lrelu(x, 0.2)

                channel = channel // 2

            # uncondition
            uncond_logit = conv(x, channels=1, kernel=4, stride=4, use_bias=True, sn=self.sn, scope='uncond_logit')

            # condition
            h_c_code = tf.concat([x, sent_emb], axis=-1)
            h_c_code = conv(h_c_code, channel, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='conv_code')
            h_c_code = batch_norm(h_c_code, is_training, scope='batch_norm_code')
            h_c_code = lrelu(h_c_code, 0.2)

            cond_logit = conv(h_c_code, channels=1, kernel=4, stride=4, use_bias=True, sn=self.sn, scope='cond_logit')

            uncond_logits.append(uncond_logit)
            cond_logits.append(cond_logit)


        return uncond_logits, cond_logits

    ##################################################################################
    # Model
    ##################################################################################


    def build_model(self):
        """ Input Image"""
        img_data_class = Image_data(self.img_height, self.img_width, self.img_ch, self.dataset_path, self.augment_flag)
        train_captions, train_images, test_captions, test_images, idx_to_word, word_to_idx = img_data_class.preprocess()
        """
        train_captions: (8855, 10, 66), test_captions: (2933, 10, 66)
        train_images: (8855,), test_images: (2933,)
        idx_to_word : 5450 5450
        """

        if self.phase == 'train' :
            self.lr = tf.placeholder(tf.float32, name='learning_rate')

            self.dataset_num = len(train_images)


            img_and_caption = tf.data.Dataset.from_tensor_slices((train_images, train_captions))

            gpu_device = '/gpu:0'
            img_and_caption = img_and_caption.apply(shuffle_and_repeat(self.dataset_num)).apply(
                map_and_batch(img_data_class.image_processing, batch_size=self.batch_size, num_parallel_batches=16,
                              drop_remainder=True)).apply(prefetch_to_device(gpu_device, None))


            img_and_caption_iterator = img_and_caption.make_one_shot_iterator()
            real_img_256, caption = img_and_caption_iterator.get_next()
            target_sentence_index = tf.random_uniform(shape=[], minval=0, maxval=10, dtype=tf.int32)
            caption = tf.gather(caption, target_sentence_index, axis=1)

            word_emb, sent_emb, mask = self.rnn_encoder(caption, n_words=len(idx_to_word),
                                                        embed_dim=self.embed_dim, drop_rate=0.5, n_hidden=128, n_layers=1,
                                                        bidirectional=True, rnn_type='lstm')

            noise = tf.random_normal(shape=[self.batch_size, self.z_dim], mean=0.0, stddev=1.0)
            fake_imgs, _, mu, logvar = self.generator(noise, sent_emb, word_emb, mask)

            real_img_64, real_img_128 = resize(real_img_256, target_size=[64, 64]), resize(real_img_256, target_size=[128, 128])
            fake_img_64, fake_img_128, fake_img_256 = fake_imgs[0], fake_imgs[1], fake_imgs[2]

            uncond_real_logits, cond_real_logits = self.discriminator([real_img_64, real_img_128, real_img_256], sent_emb)
            uncond_fake_logits, cond_fake_logits = self.discriminator([fake_img_64, fake_img_128, fake_img_256], sent_emb)

            self.g_adv_loss, self.d_adv_loss = 0, 0
            for i in range(3):
                self.g_adv_loss += self.adv_weight * (generator_loss(self.gan_type, uncond_fake_logits[i]) + generator_loss(self.gan_type, cond_fake_logits[i]))
                self.d_adv_loss += self.adv_weight * (discriminator_loss(self.gan_type, uncond_real_logits[i], uncond_fake_logits[i]) + discriminator_loss(self.gan_type, cond_real_logits[i], cond_fake_logits[i])) / 2

            self.g_kl_loss = self.kl_weight * kl_loss(mu, logvar)

            self.g_loss = self.g_adv_loss + self.g_kl_loss
            self.d_loss = self.d_adv_loss

            self.real_img = real_img_256
            self.fake_img = fake_img_256


            """ Training """
            t_vars = tf.trainable_variables()
            G_vars = [var for var in t_vars if 'generator' in var.name]
            D_vars = [var for var in t_vars if 'discriminator' in var.name]

            self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.g_loss, var_list=G_vars)
            self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.d_loss,var_list=D_vars)


            """" Summary """
            self.summary_g_loss = tf.summary.scalar("g_loss", self.g_loss)
            self.summary_d_loss = tf.summary.scalar("d_loss", self.d_loss)

            self.summary_g_adv_loss = tf.summary.scalar("g_adv_loss", self.g_adv_loss)
            self.summary_g_kl_loss = tf.summary.scalar("g_kl_loss", self.g_kl_loss)

            self.summary_d_adv_loss = tf.summary.scalar("d_adv_loss", self.d_adv_loss)


            g_summary_list = [self.summary_g_loss,
                              self.summary_g_adv_loss, self.summary_g_kl_loss]

            d_summary_list = [self.summary_d_loss,
                              self.summary_d_adv_loss]

            self.summary_merge_g_loss = tf.summary.merge(g_summary_list)
            self.summary_merge_d_loss = tf.summary.merge(d_summary_list)

        else :
            """ Test """
            self.dataset_num = len(test_captions)

            gpu_device = '/gpu:0'
            img_and_caption = tf.data.Dataset.from_tensor_slices((test_images, test_captions))

            img_and_caption = img_and_caption.apply(
                shuffle_and_repeat(self.dataset_num)).apply(
                map_and_batch(img_data_class.image_processing, batch_size=self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(
                prefetch_to_device(gpu_device, None))

            img_and_caption_iterator = img_and_caption.make_one_shot_iterator()
            real_img_256, caption = img_and_caption_iterator.get_next()
            target_sentence_index = tf.random_uniform(shape=[], minval=0, maxval=10, dtype=tf.int32)
            caption = tf.gather(caption, target_sentence_index, axis=1)

            word_emb, sent_emb, mask = self.rnn_encoder(caption, n_words=len(idx_to_word),
                                                        embed_dim=self.embed_dim, drop_rate=0.5, n_hidden=128,
                                                        n_layers=1,
                                                        bidirectional=True, rnn_type='lstm')

            noise = tf.random_normal(shape=[self.batch_size, self.z_dim], mean=0.0, stddev=1.0)
            fake_imgs, _, _, _ = self.generator(noise, sent_emb, word_emb, mask)

            self.test_real_img = real_img_256
            self.test_fake_img = fake_imgs[2]


    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=10)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_batch_id = checkpoint_counter
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")

        else:
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()

        lr = self.init_lr
        for idx in range(start_batch_id, self.iteration):

            if self.decay_flag :
                if idx > 0 and (idx % self.decay_iter) == 0 :
                    lr = self.init_lr * pow(0.5, idx // self.decay_iter)

            train_feed_dict = {
                self.lr : lr
            }

            # Update D
            _, d_loss, summary_str = self.sess.run([self.d_optim, self.d_loss, self.summary_merge_d_loss], feed_dict=train_feed_dict)
            self.writer.add_summary(summary_str, counter)

            # Update G
            real_images, fake_images, _, g_loss, summary_str = self.sess.run(
                [self.real_img, self.fake_img,
                 self.g_optim,
                 self.g_loss, self.summary_merge_g_loss], feed_dict=train_feed_dict)

            self.writer.add_summary(summary_str, counter)


            # display training status
            counter += 1
            print("Iteration: [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (idx, self.iteration, time.time() - start_time, d_loss, g_loss))

            if np.mod(idx + 1, self.print_freq) == 0:
                real_images = real_images[:5]
                fake_images = fake_images[:5]

                merge_real_images = np.expand_dims(return_images(real_images, [5, 1]), axis=0)
                merge_fake_images = np.expand_dims(return_images(fake_images, [5, 1]), axis=0)

                merge_images = np.concatenate([merge_real_images, merge_fake_images], axis=0)

                save_images(merge_images, [1, 2],
                            './{}/merge_{:07d}.jpg'.format(self.sample_dir, idx + 1))


            if np.mod(counter - 1, self.save_freq) == 0:
                self.save(self.checkpoint_dir, counter)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        if self.sn:
            sn = '_sn'
        else:
            sn = ''

        return "{}_{}_{}_{}adv_{}kl{}".format(self.model_name, self.dataset_name, self.gan_type,
                                                           self.adv_weight, self.kl_weight,
                                                           sn)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparisondkssjg
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>content</th><th>style</th><th>output</th></tr>")

        real_images, fake_images = self.sess.run([self.test_real_img, self.test_fake_img])
        for i in range(5) :
            real_path = os.path.join(self.result_dir, 'real_{}.jpg'.format(i))
            fake_path = os.path.join(self.result_dir, 'fake_{}.jpg'.format(i))

            real_image = np.expand_dims(real_images[i], axis=0)
            fake_image = np.expand_dims(fake_images[i], axis=0)

            save_images(real_image, [1, 1], real_path)
            save_images(fake_image, [1, 1], fake_path)

            index.write("<td>%s</td>" % os.path.basename(real_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (real_path if os.path.isabs(real_path) else (
                    '../..' + os.path.sep + real_path), self.img_width, self.img_height))

            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (fake_path if os.path.isabs(fake_path) else (
                    '../..' + os.path.sep + fake_path), self.img_width, self.img_height))
            index.write("</tr>")

        index.close()