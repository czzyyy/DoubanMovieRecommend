# -*-coding:utf-8-*-
import tensorflow as tf
import pickle
import os
import math
import random


class MovieRecommendModel(object):
    def __init__(self, batch_size, learning_rate, embedding_dim, training_epochs, save_dir, train_data_path):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.save_dir = save_dir
        self.train_data_path = train_data_path
        self.training_epochs = training_epochs
        self.feature_dim = None
        self.chunk_size = None
        self.train_movie_data_x = None
        self.train_movie_data_y = None
        self.columns = None
        self.means = None
        self.stds = None

    def _load_movie_raw_data(self):
        with open(self.train_data_path, 'rb') as f:
            train_movie_data = pickle.load(f)
        self.train_movie_data_x = train_movie_data['movie_train_x']
        self.train_movie_data_y = train_movie_data['movie_train_y']
        # 用于测试的时候对数据进行标准化
        self.means = train_movie_data['means']
        self.stds = train_movie_data['stds']
        self.columns = train_movie_data['columns']
        self.chunk_size = int(math.ceil(float(len(self.train_movie_data_x)) / float(self.batch_size)))
        self.feature_dim = len(self.train_movie_data_x[0])
        print('feature_dim', self.feature_dim)
        print('Chunk_size:', self.chunk_size)
        print('Full dataset tensor x:', self.train_movie_data_x.shape)
        print('Full dataset tensor y:', self.train_movie_data_y.shape)

    def _get_batch_raw_data(self, index):
        x_batch = self.train_movie_data_x[index: index + self.batch_size]
        y_batch = self.train_movie_data_y[index: index + self.batch_size]
        return x_batch, y_batch

    def _full_connect(self, x, output_num, stddev=0.02, bias=0.0, name='full_connect', reuse=False):
        """
        :param x: the input feature map
        :param output_num: the output feature map size
        :param stddev:
        :param bias:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name, reuse=reuse):
            shape = x.shape.as_list()
            w = tf.get_variable('w', [shape[1], output_num], tf.float32, tf.truncated_normal_initializer(stddev=stddev))
            b = tf.get_variable('b', [output_num], tf.float32, tf.constant_initializer(bias))
            return tf.matmul(x, w) + b

    def _batch_normalizer(self, x, epsilon=1e-5, momentum=0.9, train=True, name='batch_norm', reuse=False):
        #  inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
        #     The normalization is over all but the last dimension if `data_format` is `NHWC` and the
        #     second dimension if `data_format` is `NCHW`
        """
        :param x: input feature map
        :param epsilon:
        :param momentum:
        :param train: train or not?
        :param name:
        :param reuse: reuse or not?
        :return:
        """
        with tf.variable_scope(name, reuse=reuse):
            return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon,
                                                scale=True, is_training=train)

    def _mse_loss(self, pred, data):
        loss_val = tf.reduce_mean(tf.multiply((pred - data), (pred - data))) / 2.0
        return loss_val

    def _movie_net(self, x, keep_prob, train=True, reuse=False, name='movie_net'):
        with tf.variable_scope(name, reuse=reuse):
            full1 = self._full_connect(x, output_num=512, name='full1', reuse=reuse)
            full1 = tf.nn.leaky_relu(full1, name='lrelu1')
            full1 = tf.nn.dropout(full1, keep_prob=keep_prob, name='d')

            full2 = self._full_connect(full1, output_num=256, name='full2', reuse=reuse)
            full2 = tf.squeeze(
                self._batch_normalizer(tf.expand_dims(full2, axis=-1), train=train, name='b2', reuse=reuse), axis=-1)
            full2 = tf.nn.leaky_relu(full2, name='lrelu2')

            full3 = self._full_connect(full2, output_num=125, name='full3', reuse=reuse)
            full3 = tf.squeeze(
                self._batch_normalizer(tf.expand_dims(full3, axis=-1), train=train, name='b3', reuse=reuse), axis=-1)
            full3 = tf.nn.leaky_relu(full3, name='lrelu3')

            full4 = self._full_connect(full3, output_num=self.embedding_dim, name='full4', reuse=reuse)
            full4 = tf.squeeze(
                self._batch_normalizer(tf.expand_dims(full4, axis=-1), train=train, name='b4', reuse=reuse), axis=-1)
            embedding = tf.nn.leaky_relu(full4, name='lrelu4')  # [None, 64]

            full5 = self._full_connect(embedding, output_num=1, name='full5', reuse=reuse)
            output = tf.nn.sigmoid(full5, name='output')
            return embedding, output

    def train(self):
        # 载入数据
        self._load_movie_raw_data()
        with tf.name_scope('inputs'):
            input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_dim], name='input_x')
            input_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='input_y')
            dropout_prob = tf.placeholder(dtype=tf.float32, name='dropout_prob')
        with tf.name_scope('loss'):
            embedding_x, pred_y = self._movie_net(input_x, dropout_prob, train=True, reuse=False)
            loss = self._mse_loss(pred_y, input_y)
            tf.summary.scalar('loss', loss)
        with tf.name_scope('optimizer'):
            trainer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        with tf.name_scope('train'):
            with tf.Session() as sess:
                saver = tf.train.Saver()
                # merge summary
                merged = tf.summary.merge_all()
                # choose dir
                writer = tf.summary.FileWriter(self.save_dir + 'movie', sess.graph)
                sess.run(tf.global_variables_initializer())

                batch_index = 0
                for e in range(self.training_epochs):
                    for batch_i in range(self.chunk_size):
                        batch_index = (batch_index + self.batch_size) % ((self.chunk_size - 1) * self.batch_size)
                        x_batch, y_batch = self._get_batch_raw_data(batch_index)
                        sess.run(trainer, feed_dict={input_x: x_batch, input_y: y_batch, dropout_prob: 0.7})
                        if (self.chunk_size * e + batch_i) % 512 == 0:
                            train_loss = sess.run(loss, feed_dict={input_x: x_batch, input_y: y_batch, dropout_prob: 1.0})

                            merge_result = sess.run(merged,
                                                    feed_dict={input_x: x_batch, input_y: y_batch, dropout_prob: 1.0})
                            writer.add_summary(merge_result, self.chunk_size * e + batch_i)

                            print(
                                "step {} of epoch {}/{}".format(self.chunk_size * e + batch_i, e, self.training_epochs),
                                "Loss: {:.4f}".format(train_loss))
                print('train done')
                # save sess
                saver.save(sess, self.save_dir + 'movie/movie.ckpt')

    def predict_score(self, predict_movie_datas, norm=False):
        if norm:
            predict_movie_datas[:, self.columns] = (predict_movie_datas[:, self.columns] - self.means) / self.stds
        input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_dim], name='input_x')
        dropout_prob = tf.placeholder(dtype=tf.float32, name='dropout_prob')
        embedding_x, pred_y = self._movie_net(input_x, dropout_prob, train=False, reuse=False)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            ckpt_path = self.save_dir + 'movie/movie.ckpt'
            if ckpt_path:
                print('Model loaded from {}....start'.format(ckpt_path))
                for var in vars_list:
                    vname = var.name
                    from_name = vname
                    print(from_name)
                    var_value = tf.contrib.framework.load_variable(ckpt_path, from_name)
                    assign_ops.append(tf.assign(var, var_value))
                sess.run(assign_ops)
                print('Model loaded from {}....end'.format(ckpt_path))
            else:
                print('Model loading is fail')
            result_embedding, result_y = sess.run([embedding_x, pred_y],
                                                  feed_dict={input_x: predict_movie_datas, dropout_prob: 1.0})
            # 返回该新电影的 特征嵌入 以及 评分
            return result_embedding, result_y * 10.0

    def save_train_movie_embedding(self):
        input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_dim], name='input_x')
        dropout_prob = tf.placeholder(dtype=tf.float32, name='dropout_prob')
        embedding_x, pred_y = self._movie_net(input_x, dropout_prob, train=False, reuse=True)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.save_dir + 'movie/movie.ckpt')
            result_embedding, result_y = sess.run([embedding_x, pred_y],
                                                  feed_dict={input_x: self.train_movie_data_x, dropout_prob: 1.0})

            if not os.path.exists(self.save_dir + 'movie/train_movie_embedding.pickle'):
                open(self.save_dir + 'movie/train_movie_embedding.pickle', 'w')
                print('train_movie_embedding.pickle not exists, create it')
            with open(self.save_dir + 'movie/train_movie_embedding.pickle', 'wb') as f:
                pickle.dump(result_embedding, f)
            print('save train_movie_embedding.pickle end')
            return result_y


class MovieUserRecommendModel(object):
    def __init__(self, batch_size, learning_rate, embedding_dim, training_epochs, save_dir, user_data_path,
                 movie_data_path, movie_embedding_data_path):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.save_dir = save_dir
        self.user_data_path = user_data_path
        self.movie_data_path = movie_data_path
        self.movie_embedding_data_path = movie_embedding_data_path
        self.training_epochs = training_epochs
        self.user_feature_dim = None
        self.movie_feature_dim = None
        self.movie_embedding_feature_dim = None
        self.chunk_size = None
        self.train_user_data_x = None
        self.train_user_data_y = None
        self.movie_data = None
        self.movie_embedding_data = None
        self.columns = None
        self.means = None
        self.stds = None
        self.user_num = None
        self.user_embedding = None

    def _load_movie_user_raw_data(self):
        with open(self.movie_data_path, 'rb') as f:
            movie_data = pickle.load(f)
        self.movie_data = movie_data['movie_train_x']

        with open(self.movie_embedding_data_path, 'rb') as f:
            self.movie_embedding_data = pickle.load(f)

        with open(self.user_data_path, 'rb') as f:
            train_user_data = pickle.load(f)
        self.train_user_data_x = train_user_data['user_train_x']
        self.train_user_data_y = train_user_data['user_train_y']
        self.user_num = train_user_data['user_num']
        # 打乱用户数据
        total_train_index = list(range(len(self.train_user_data_x)))
        random.shuffle(total_train_index)
        self.train_user_data_x = self.train_user_data_x[total_train_index]
        self.train_user_data_y = self.train_user_data_y[total_train_index]
        # 用于测试的时候对数据进行标准化
        self.means = train_user_data['means']
        self.stds = train_user_data['stds']
        self.columns = train_user_data['columns']
        self.chunk_size = int(math.ceil(float(len(self.train_user_data_x)) / float(self.batch_size)))
        self.user_feature_dim = len(self.train_user_data_x[0]) - 2  # 因为最后两位是电影索引 和 用户索引
        self.movie_feature_dim = len(self.movie_data[0])
        self.movie_embedding_feature_dim = len(self.movie_embedding_data[0])
        print('Chunk_size:', self.chunk_size)
        print('Full dataset tensor x:', self.train_user_data_x.shape)
        print('Full dataset tensor y:', self.train_user_data_y.shape)

    def _get_batch_raw_data(self, index):
        batch = self.train_user_data_x[index: index + self.batch_size]
        user_y_batch = self.train_user_data_y[index: index + self.batch_size]
        user_x_batch = batch[:, self.columns]  # 除了最后两列
        movie_index = [int(b) for b in batch[:, -2]]
        movie_batch = self.movie_data[movie_index]  # 倒数第二列是电影索引
        movie_embedding_batch = self.movie_embedding_data[movie_index]
        user_index_batch = [int(b) for b in batch[:, -1]]  # 最后一列是用户索引
        return user_x_batch, user_y_batch, movie_batch, movie_embedding_batch, user_index_batch

    def _full_connect(self, x, output_num, stddev=0.02, bias=0.0, name='full_connect', reuse=False):
        """
        :param x: the input feature map
        :param output_num: the output feature map size
        :param stddev:
        :param bias:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name, reuse=reuse):
            shape = x.shape.as_list()
            w = tf.get_variable('w', [shape[1], output_num], tf.float32, tf.truncated_normal_initializer(stddev=stddev))
            b = tf.get_variable('b', [output_num], tf.float32, tf.constant_initializer(bias))
            return tf.matmul(x, w) + b

    def _batch_normalizer(self, x, epsilon=1e-5, momentum=0.9, train=True, name='batch_norm', reuse=False):
        #  inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
        #     The normalization is over all but the last dimension if `data_format` is `NHWC` and the
        #     second dimension if `data_format` is `NCHW`
        """
        :param x: input feature map
        :param epsilon:
        :param momentum:
        :param train: train or not?
        :param name:
        :param reuse: reuse or not?
        :return:
        """
        with tf.variable_scope(name, reuse=reuse):
            return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon,
                                                scale=True, is_training=train)

    def _mse_loss(self, pred, data):
        loss_val = tf.reduce_mean(tf.multiply((pred - data), (pred - data))) / 2.0
        return loss_val

    def _l1_loss(self, pred, data):
        return tf.reduce_mean(tf.abs(pred - data))

    def _movie_user_net(self, u_x, m_r_x, m_e_x, u_index, keep_prob, train=True, reuse=False, name='movie_user_net'):
        with tf.variable_scope(name, reuse=reuse):
            # 每个用户对应一个嵌入
            user_embeddings = tf.get_variable('uew', [self.user_num, self.embedding_dim], tf.float32,
                                              tf.truncated_normal_initializer(stddev=0.02))
            embedding = tf.nn.embedding_lookup(user_embeddings, u_index)

            full_emb = self._full_connect(tf.concat([u_x, embedding], axis=-1), output_num=512, name='full_emb', reuse=reuse)
            full_emb = tf.squeeze(
                self._batch_normalizer(tf.expand_dims(full_emb, axis=-1), train=train, name='b_emb', reuse=reuse),
                axis=-1)
            full_emb = tf.nn.elu(full_emb, name='lrelu_emb')
            full_emb = tf.nn.dropout(full_emb, keep_prob=keep_prob, name='drop_emb')

            full_emb2 = self._full_connect(tf.concat([m_r_x, m_e_x], axis=-1), output_num=512, name='full_emb2',
                                           reuse=reuse)
            full_emb2 = tf.squeeze(
                self._batch_normalizer(tf.expand_dims(full_emb2, axis=-1), train=train, name='b_emb2', reuse=reuse),
                axis=-1)
            full_emb2 = tf.nn.elu(full_emb2, name='lrelu_emb2')
            full_emb2 = tf.nn.dropout(full_emb2, keep_prob=keep_prob, name='drop_emb2')

            full2 = self._full_connect(tf.concat([full_emb, full_emb2], axis=-1), output_num=256, name='full2', reuse=reuse)
            full2 = tf.squeeze(
                self._batch_normalizer(tf.expand_dims(full2, axis=-1), train=train, name='b2', reuse=reuse), axis=-1)
            full2 = tf.nn.elu(full2, name='lrelu2')

            full3 = self._full_connect(full2, output_num=128, name='full3', reuse=reuse)
            full3 = tf.squeeze(
                self._batch_normalizer(tf.expand_dims(full3, axis=-1), train=train, name='b3', reuse=reuse), axis=-1)
            full3 = tf.nn.elu(full3, name='lrelu3')

            full4 = self._full_connect(full3, output_num=64, name='full4', reuse=reuse)
            full4 = tf.squeeze(
                self._batch_normalizer(tf.expand_dims(full4, axis=-1), train=train, name='b4', reuse=reuse), axis=-1)
            full4 = tf.nn.elu(full4, name='lrelu4')

            full5 = self._full_connect(full4, output_num=1, name='full5', reuse=reuse)
            output = tf.nn.elu(full5, name='output')
            return user_embeddings, output

    def train(self):
        # 载入数据
        self._load_movie_user_raw_data()
        with tf.name_scope('inputs'):
            user_input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.user_feature_dim], name='user_input_x')
            user_input_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='user_input_y')
            movie_input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.movie_feature_dim], name='movie_input_x')
            movie_e_input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.movie_embedding_feature_dim]
                                             , name='movie_embedding_input_x')
            user_index_input = tf.placeholder(dtype=tf.int32, shape=[None], name='user_index_input')
            dropout_prob = tf.placeholder(dtype=tf.float32, name='dropout_prob')
        with tf.name_scope('loss'):
            pred_user_embeddings, pred_y = self._movie_user_net(user_input_x, movie_input_x, movie_e_input_x,
                                                                user_index_input, dropout_prob, train=True, reuse=False)
            loss = self._mse_loss(pred_y, user_input_y)
            # loss = self._l1_loss(pred_y, user_input_y)
            tf.summary.scalar('loss', loss)
        with tf.name_scope('optimizer'):
            trainer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        with tf.name_scope('train'):
            with tf.Session() as sess:
                saver = tf.train.Saver()
                # merge summary
                merged = tf.summary.merge_all()
                # choose dir
                writer = tf.summary.FileWriter(self.save_dir + 'movie_user', sess.graph)
                sess.run(tf.global_variables_initializer())

                batch_index = 0
                for e in range(self.training_epochs):
                    for batch_i in range(self.chunk_size):
                        batch_index = (batch_index + self.batch_size) % ((self.chunk_size - 1) * self.batch_size)
                        u_x_batch, u_y_batch, m_batch, m_e_batch, u_i_batch = self._get_batch_raw_data(batch_index)
                        sess.run(trainer, feed_dict={user_input_x: u_x_batch, user_input_y: u_y_batch,
                                                     movie_input_x: m_batch, movie_e_input_x: m_e_batch,
                                                     user_index_input: u_i_batch, dropout_prob: 0.7})
                        train_em = sess.run(pred_user_embeddings)
                        if (self.chunk_size * e + batch_i) % 1024 == 0:
                            train_loss = sess.run(loss, feed_dict={user_input_x: u_x_batch, user_input_y: u_y_batch,
                                                                   movie_input_x: m_batch,
                                                                   movie_e_input_x: m_e_batch,
                                                                   user_index_input: u_i_batch, dropout_prob: 1.0})

                            merge_result = sess.run(merged,
                                                    feed_dict={user_input_x: u_x_batch, user_input_y: u_y_batch,
                                                               movie_input_x: m_batch, movie_e_input_x: m_e_batch,
                                                               user_index_input: u_i_batch, dropout_prob: 1.0})
                            writer.add_summary(merge_result, self.chunk_size * e + batch_i)
                            print(
                                "step {} of epoch {}/{}".format(self.chunk_size * e + batch_i, e, self.training_epochs),
                                "Loss: {:.4f}".format(train_loss))
                self.user_embedding = train_em
                print('train done')
                # save sess
                saver.save(sess, self.save_dir + 'movie_user/movie_user.ckpt')

    def predict_score(self, user_predict, movie_predict, movie_e_predict, user_predict_index):
        # 默认电影的数据都是经过标准化的
        user_input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.user_feature_dim], name='user_input_x')
        movie_input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.movie_feature_dim], name='movie_input_x')
        movie_e_input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.movie_embedding_feature_dim]
                                         , name='movie_embedding_input_x')
        user_index_input = tf.placeholder(dtype=tf.int32, shape=[None], name='user_index_input')
        dropout_prob = tf.placeholder(dtype=tf.float32, name='dropout_prob')
        pred_y = self._movie_user_net(user_input_x, movie_input_x, movie_e_input_x, user_index_input, dropout_prob,
                                      train=False, reuse=False)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            ckpt_path = self.save_dir + 'movie_user/movie_user.ckpt'
            if ckpt_path:
                print('Model loaded from {}....start'.format(ckpt_path))
                for var in vars_list:
                    vname = var.name
                    from_name = vname
                    print(from_name)
                    var_value = tf.contrib.framework.load_variable(ckpt_path, from_name)
                    assign_ops.append(tf.assign(var, var_value))
                sess.run(assign_ops)
                print('Model loaded from {}....end'.format(ckpt_path))
            else:
                print('Model loading is fail')

            _, result_y = sess.run(pred_y, feed_dict={user_input_x: user_predict, movie_input_x: movie_predict,
                                                      movie_e_input_x: movie_e_predict,
                                                      user_index_input: user_predict_index, dropout_prob: 1.0})
            return result_y * 10.0

    def save_train_user_embedding(self):
        print(self.user_embedding[0][0:10])
        if not os.path.exists(self.save_dir + 'movie_user/train_user_embedding.pickle'):
            open(self.save_dir + 'movie_user/train_user_embedding.pickle', 'w')
            print('train_user_embedding.pickle not exists, create it')
        with open(self.save_dir + 'movie_user/train_user_embedding.pickle', 'wb') as f:
            pickle.dump(self.user_embedding, f)
        print(self.user_embedding.shape)
        print('save train_user_embedding.pickle end')


if __name__ == '__main__':
    save_d = './douban/my_data/movie_user_noraw/'
    movie_data_p = './douban/my_data/train_data/train_movie_data.pickle'
    user_data_p = './douban/my_data/train_data/train_user_data.pickle'
    movie_embed_p = save_d + 'movie/train_movie_embedding.pickle'
    movie_model = MovieRecommendModel(batch_size=64, learning_rate=0.01, embedding_dim=64, training_epochs=100,
                                      save_dir=save_d, train_data_path=movie_data_p)
    movie_model.train()
    movie_model.save_train_movie_embedding()

    # user_model = MovieUserRecommendModel(batch_size=64, learning_rate=0.001, embedding_dim=24, training_epochs=100,
    #                                      save_dir=save_d, user_data_path=user_data_p,
    #                                      movie_data_path=movie_data_p, movie_embedding_data_path=movie_embed_p)
    # user_model.train()
    # user_model.save_train_user_embedding()