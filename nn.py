# -*- coding: utf-8 -*-

"""
神经网络模型
"""

import logging
import tensorflow as tf
import random


class NNet:
    def __init__(self, layer_size):
        '''
        :param layer_size: 从输入层到输出层每层神经元个数
        '''

        # 以下构建神经网络模型
        # 每个对象拥有一个私有的tensorflow Graph，所有tensorflow操作在其下进行
        self.graph = tf.Graph()
        with self.graph.as_default():
            if len(layer_size) < 3:
                logging.error('neural network layer number %g less than 3'
                              % len(layer_size))
                raise RuntimeError('neural network layer number %g less than 3'
                                   % len(layer_size))

            # Build model structure
            # tf Graph input
            # x,y在run之前必须fed值，不然会报错
            self.x = tf.placeholder("float", [None, layer_size[0]])
            # list[-1]指向最后一个元素
            self.y = tf.placeholder("float", [None, layer_size[-1]])

            # layers weight & bias
            self.weights, self.biases = [], []
            for i in range(len(layer_size) - 1):
                self.weights.append(tf.Variable(tf.random_normal(
                                    [layer_size[i], layer_size[i + 1]])))
                self.biases.append(tf.Variable(tf.random_normal(
                                   [layer_size[i + 1]])))

            # Hidden layer with RELU activation
            hidden_layer = tf.add(tf.matmul(self.x, self.weights[0]),
                                  self.biases[0])
            # Computes rectified linear: `max(hidden_layer, 0)'
            hidden_layer = tf.nn.relu(hidden_layer)
            for i in range(1, len(self.weights) - 1):
                hidden_layer = tf.add(tf.matmul(hidden_layer,
                                      self.weights[i]), self.biases[i])
                hidden_layer = tf.nn.relu(hidden_layer)
            # Output layer is a linear layer
            self.out = tf.add(tf.matmul(hidden_layer, self.weights[-1]),
                              self.biases[-1])

    def train(self, train_x, train_y, model_path=None, learning_rate=0.001,
              training_epochs=15, display_step=1):
        '''
        模型训练
        :param model_path: 模型保存路径
        :param learning_rate: 学习速率
        :param training_epochs: 训练循环次数
        :param display_step: 每display_step步输出状态
        '''
        logging.info('train neural network with %g sample(s)' % len(train_x))

        with self.graph.as_default():
            # Define loss and optimizer
            cost = tf.reduce_mean(tf.square(self.out - self.y))
            # learning_rate=learning_rate->learning_rate
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
            saver = tf.train.Saver()

            # Launch the graph
            with tf.Session() as sess:
                # Initializing the variables
                sess.run(tf.global_variables_initializer())

                # Training cycle
                for epoch in range(training_epochs):
                    avg_cost = 0.
                    for _ in range(len(train_x)):
                        i = random.randint(0, len(train_x) - 1)
                        # Run optimization op (backprop) and cost op
                        # (to get loss value)
                        _, c = sess.run([optimizer, cost], feed_dict={self.x:
                                        train_x[i], self.y: train_y[i]})
                        # Compute average loss
                        avg_cost += c / len(train_x)

                    # Display logs per display_step epoch step
                    if epoch % display_step == 0:
                        logging.info("Epoch: %04d cost= %.9f"
                                     % ((epoch + 1), avg_cost))
                    if model_path is not None:
                        saver.save(sess, model_path)
                logging.info("Optimization Finished!")

    def test(self, test_x, test_y, model_path=None):
        logging.info('test neural network with %g sample(s)' % len(test_x))

        with self.graph.as_default():
            cost = tf.reduce_mean(tf.square(self.out - self.y))

            saver = tf.train.Saver()

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                if model_path is not None:
                    saver.restore(sess, model_path)

                avg_cost = 0.
                for one_x, one_y in zip(test_x, test_y):
                    c = cost.eval({self.x: one_x, self.y: one_y})
                    # Compute average loss
                    avg_cost += c / len(test_x)

            logging.info("cost= %.9f" % avg_cost)

    def run(self, run_x, model_path=None):
        '''
        运行模型得到预测
        :param run_x: 输入列表 [[...]...]
        :param model_path: 模型保存路径
        :return: 输出列表
        '''
        with self.graph.as_default():
            saver = tf.train.Saver()

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                if model_path is not None:
                    saver.restore(sess, model_path)

                result = []
                for rx in run_x:
                    result.append(self.out.eval({self.x: rx}))
                return result
