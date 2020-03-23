from utils import *
import time
from tensorflow.python.data.experimental import prefetch_to_device, shuffle_and_repeat, map_and_batch # >= tf 1.15
from networks import *

class AttnGAN():
    def __init__(self, args):

        self.phase = args.phase
        self.model_name = 'AttnGAN'

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

        self.d_dim = args.d_dim
        self.g_dim = args.g_dim
        self.embed_dim = args.embed_dim
        self.z_dim = args.z_dim


        """ Weight """
        self.adv_weight = args.adv_weight
        self.kl_weight = args.kl_weight
        self.embed_weight = args.embed_weight



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
        print("# embed_weight : ", self.embed_weight)

        print()

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Input Image"""
        img_data_class = Image_data(self.img_height, self.img_width, self.img_ch, self.dataset_path, self.augment_flag)
        train_class_id, train_captions, train_images, test_captions, test_images, idx_to_word, word_to_idx = img_data_class.preprocess()
        self.vocab_size = len(idx_to_word)
        self.idx_to_word = idx_to_word
        self.word_to_idx = word_to_idx
        """
        train_captions: (8855, 10, 18), test_captions: (2933, 10, 18)
        train_images: (8855,), test_images: (2933,)
        idx_to_word : 5450 5450
        """

        if self.phase == 'train' :
            self.dataset_num = len(train_images)

            img_and_caption = tf.data.Dataset.from_tensor_slices((train_images, train_captions, train_class_id))

            gpu_device = '/gpu:0'
            img_and_caption = img_and_caption.apply(shuffle_and_repeat(self.dataset_num)).apply(
                map_and_batch(img_data_class.image_processing, batch_size=self.batch_size, num_parallel_batches=16,
                              drop_remainder=True)).apply(prefetch_to_device(gpu_device, None))

            self.img_caption_iter = iter(img_and_caption)
            # real_img_256, caption = iter(img_and_caption)

            """ Network """
            self.rnn_encoder = RnnEncoder(n_words=self.vocab_size, embed_dim=self.embed_dim,
                                          drop_rate=0.5, n_hidden=128, n_layer=1,
                                          bidirectional=True, rnn_type='lstm')
            self.cnn_encoder = CnnEncoder(embed_dim=self.embed_dim)

            self.ca_net = CA_NET(c_dim=self.z_dim)
            self.generator = Generator(channels=self.g_dim)

            self.discriminator = Discriminator(channels=self.d_dim, embed_dim=self.embed_dim)

            """ Optimizer """
            self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.init_lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08)

            d_64_optimizer = tf.keras.optimizers.Adam(learning_rate=self.init_lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
            d_128_optimizer = tf.keras.optimizers.Adam(learning_rate=self.init_lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
            d_256_optimizer = tf.keras.optimizers.Adam(learning_rate=self.init_lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
            self.d_optimizer = [d_64_optimizer, d_128_optimizer, d_256_optimizer]

            self.embed_optimizer = tf.keras.optimizers.Adam(learning_rate=self.init_lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08)


            """ Checkpoint """
            self.ckpt = tf.train.Checkpoint(rnn_encoder=self.rnn_encoder, cnn_encoder=self.cnn_encoder,
                                            ca_net=self.ca_net,
                                            generator=self.generator,
                                            discriminator=self.discriminator,
                                            g_optimizer=self.g_optimizer,
                                            d_64_optimizer=d_64_optimizer,
                                            d_128_optimizer=d_128_optimizer,
                                            d_256_optimizer=d_256_optimizer,
                                            embed_optimizer=self.embed_optimizer)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=2)
            self.start_iteration = 0

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint)
                self.start_iteration = int(self.manager.latest_checkpoint.split('-')[-1])
                print('Latest checkpoint restored!!')
                print('start iteration : ', self.start_iteration)
            else:
                print('Not restoring from saved checkpoint')

        else :
            """ Test """
            self.dataset_num = len(test_captions)

            gpu_device = '/gpu:0'
            img_and_caption = tf.data.Dataset.from_tensor_slices((test_images, test_captions))

            img_and_caption = img_and_caption.apply(
                shuffle_and_repeat(self.dataset_num)).apply(
                map_and_batch(img_data_class.image_processing, batch_size=self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(
                prefetch_to_device(gpu_device, None))

            self.img_caption_iter = iter(img_and_caption)

            """ Network """
            self.rnn_encoder = RnnEncoder(n_words=self.vocab_size, embed_dim=self.embed_dim,
                                          drop_rate=0.5, n_hidden=128, n_layer=1,
                                          bidirectional=True, rnn_type='lstm')
            self.ca_net = CA_NET(c_dim=self.z_dim)
            self.generator = Generator(channels=self.g_dim)


            """ Checkpoint """
            self.ckpt = tf.train.Checkpoint(rnn_encoder=self.rnn_encoder,
                                            ca_net = self.ca_net,
                                            generator=self.generator)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=2)
            self.start_iteration = 0

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
                print('Latest checkpoint restored!!')
                print('start iteration : ', self.start_iteration)
            else:
                print('Not restoring from saved checkpoint')

    def embed_train_step(self, real_256, caption, class_id):
        with tf.GradientTape() as embed_tape:
            target_sentence_index = tf.random.uniform(shape=[], minval=0, maxval=10, dtype=tf.int32)
            caption = tf.gather(caption, target_sentence_index, axis=1)

            word_feature, sent_code = self.cnn_encoder(real_256, training=True)
            word_emb, sent_emb, mask = self.rnn_encoder(caption, training=True)

            w_loss = word_loss(word_feature, word_emb, class_id)
            s_loss = sent_loss(sent_code, sent_emb, class_id)

            embed_loss = self.embed_weight * (w_loss + s_loss)

        embed_train_variable = self.cnn_encoder.trainable_variables + self.rnn_encoder.trainable_variables
        embed_gradient = embed_tape.gradient(embed_loss, embed_train_variable)
        self.embed_optimizer.apply_gradients(zip(embed_gradient, embed_train_variable))

        return embed_loss

    def d_train_step(self, real_256, caption):
        with tf.GradientTape() as d_64_tape, tf.GradientTape() as d_128_tape, tf.GradientTape() as d_256_tape :
            target_sentence_index = tf.random.uniform(shape=[], minval=0, maxval=10, dtype=tf.int32)
            caption = tf.gather(caption, target_sentence_index, axis=1)

            word_emb, sent_emb, mask = self.rnn_encoder(caption, training=True)
            z_code = tf.random.normal(shape=[self.batch_size, self.z_dim])
            c_code, mu, logvar = self.ca_net(sent_emb, training=True)
            fake_imgs = self.generator([c_code, z_code, word_emb, mask], training=True)

            real_64, real_128 = resize(real_256, target_size=[64, 64]), resize(real_256, target_size=[128, 128])
            fake_64, fake_128, fake_256 = fake_imgs

            uncond_real_logits, cond_real_logits = self.discriminator([real_64, real_128, real_256, sent_emb], training=True)
            _, cond_wrong_logits = self.discriminator([real_64[:(self.batch_size - 1)], real_128[:(self.batch_size - 1)], real_256[:(self.batch_size - 1)], sent_emb[1:self.batch_size]])
            uncond_fake_logits, cond_fake_logits = self.discriminator([fake_64, fake_128, fake_256, sent_emb], training=True)

            d_adv_loss = []

            for i in range(3):
                uncond_real_loss, uncond_fake_loss = discriminator_loss(self.gan_type, uncond_real_logits[i], uncond_fake_logits[i])
                cond_real_loss, cond_fake_loss = discriminator_loss(self.gan_type, cond_real_logits[i], cond_fake_logits[i])
                _, cond_wrong_loss = discriminator_loss(self.gan_type, None, cond_wrong_logits[i])

                each_d_adv_loss = self.adv_weight * (((uncond_real_loss + cond_real_loss) / 2) + (uncond_fake_loss + cond_fake_loss + cond_wrong_loss) / 3)
                d_adv_loss.append(each_d_adv_loss)

            d_loss = tf.reduce_sum(d_adv_loss)

        d_train_variable = [self.discriminator.d_64.trainable_variables,
                            self.discriminator.d_128.trainable_variables,
                            self.discriminator.d_256.trainable_variables]
        d_tape = [d_64_tape, d_128_tape, d_256_tape]

        for i in range(3):
            d_gradient = d_tape[i].gradient(d_adv_loss[i], d_train_variable[i])
            self.d_optimizer[i].apply_gradients(zip(d_gradient, d_train_variable[i]))

        return d_loss, tf.reduce_sum(d_adv_loss)

    def g_train_step(self, caption, class_id):
        with tf.GradientTape() as g_tape:
            target_sentence_index = tf.random.uniform(shape=[], minval=0, maxval=10, dtype=tf.int32)
            caption = tf.gather(caption, target_sentence_index, axis=1)

            word_emb, sent_emb, mask = self.rnn_encoder(caption, training=True)

            z_code = tf.random.normal(shape=[self.batch_size, self.z_dim])
            c_code, mu, logvar = self.ca_net(sent_emb, training=True)

            fake_imgs = self.generator([c_code, z_code, word_emb, mask], training=True)
            fake_64, fake_128, fake_256 = fake_imgs

            uncond_fake_logits, cond_fake_logits = self.discriminator([fake_64, fake_128, fake_256, sent_emb], training=True)

            g_adv_loss = 0

            for i in range(3):
                g_adv_loss += self.adv_weight * (generator_loss(self.gan_type, uncond_fake_logits[i]) + generator_loss(self.gan_type, cond_fake_logits[i]))

            word_feature, sent_code = self.cnn_encoder(fake_256, training=True)

            w_loss = word_loss(word_feature, word_emb, class_id)
            s_loss = sent_loss(sent_code, sent_emb, class_id)

            g_embed_loss = self.embed_weight * (w_loss + s_loss) * 5.0

            g_kl_loss = self.kl_weight * kl_loss(mu, logvar)

            g_loss = g_adv_loss + g_kl_loss + g_embed_loss

        g_train_variable = self.generator.trainable_variables + self.ca_net.trainable_variables
        g_gradient = g_tape.gradient(g_loss, g_train_variable)
        self.g_optimizer.apply_gradients(zip(g_gradient, g_train_variable))

        return g_loss, g_adv_loss, g_kl_loss, g_embed_loss, fake_256

    def train(self):
        start_time = time.time()

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, self.model_dir))

        for idx in range(self.start_iteration, self.iteration):
            current_step = idx
            if self.decay_flag:
                # total_step = self.iteration
                decay_start_step = self.decay_iter

                # if current_step >= decay_start_step :
                # lr = self.init_lr * (total_step - current_step) / (total_step - decay_start_step)
                if idx > 0 and (idx % decay_start_step) == 0:
                    lr = self.init_lr * pow(0.5, idx // decay_start_step)
                    self.g_optimizer.learning_rate = lr
                    for i in range(3):
                        self.d_optimizer[i].learning_rate = lr
                    self.embed_optimizer.learning_rate = lr

            real_256, caption, class_id = next(self.img_caption_iter)

            embed_loss = self.embed_train_step(real_256, caption, class_id)

            d_loss, d_adv_loss = self.d_train_step(real_256, caption)

            g_loss, g_adv_loss, g_kl_loss, g_embed_loss, fake_256 = self.g_train_step(caption, class_id)
            g_loss += embed_loss

            # save to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('g_adv_loss', g_adv_loss, step=idx)
                tf.summary.scalar('g_kl_loss', g_kl_loss, step=idx)
                tf.summary.scalar('g_embed_loss', g_embed_loss, step=idx)
                tf.summary.scalar('g_loss', g_loss, step=idx)

                tf.summary.scalar('embed_loss', embed_loss, step=idx)

                tf.summary.scalar('d_adv_loss', d_adv_loss, step=idx)
                tf.summary.scalar('d_loss', d_loss, step=idx)


            # save every self.save_freq
            if np.mod(idx + 1, self.save_freq) == 0:
                self.manager.save(checkpoint_number=idx + 1)

            if np.mod(idx + 1, self.print_freq) == 0:
                real_images = real_256[:5]
                fake_images = fake_256[:5]

                merge_real_images = np.expand_dims(return_images(real_images, [5, 1]), axis=0)
                merge_fake_images = np.expand_dims(return_images(fake_images, [5, 1]), axis=0)

                merge_images = np.concatenate([merge_real_images, merge_fake_images], axis=0)

                save_images(merge_images, [1, 2],
                            './{}/merge_{:07d}.jpg'.format(self.sample_dir, idx + 1))

            # display training status
            print("Iteration: [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (idx, self.iteration, time.time() - start_time, d_loss, g_loss))

        # save model for final step
        self.manager.save(checkpoint_number=self.iteration)

    @property
    def model_dir(self):
        if self.sn:
            sn = '_sn'
        else:
            sn = ''

        return "{}_{}_{}_{}adv_{}kl_{}embed{}".format(self.model_name, self.dataset_name, self.gan_type,
                                                           self.adv_weight, self.kl_weight, self.embed_weight,
                                                           sn)


    def test(self):

        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        real_256, caption = next(self.img_caption_iter)
        target_sentence_index = tf.random.uniform(shape=[], minval=0, maxval=10, dtype=tf.int32)
        caption = tf.gather(caption, target_sentence_index, axis=1)

        word_emb, sent_emb, mask = self.rnn_encoder(caption, training=False)

        z = tf.random.normal(shape=[self.batch_size, self.z_dim])
        fake_imgs, _, _ = self.generator([z, sent_emb, word_emb, mask], training=False)

        fake_256 = fake_imgs[-1]

        for i in range(5) :
            real_path = os.path.join(self.result_dir, 'real_{}.jpg'.format(i))
            fake_path = os.path.join(self.result_dir, 'fake_{}.jpg'.format(i))

            real_image = np.expand_dims(real_256[i], axis=0)
            fake_image = np.expand_dims(fake_256[i], axis=0)

            save_images(real_image, [1, 1], real_path)
            save_images(fake_image, [1, 1], fake_path)

            index.write("<td>%s</td>" % os.path.basename(real_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (real_path if os.path.isabs(real_path) else (
                    '../..' + os.path.sep + real_path), self.img_width, self.img_height))

            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (fake_path if os.path.isabs(fake_path) else (
                    '../..' + os.path.sep + fake_path), self.img_width, self.img_height))
            index.write("</tr>")

        index.close()