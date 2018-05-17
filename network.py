import os.path


import tensorflow as tf
from keras.layers import Input, Conv2D, regularizers, BatchNormalization, LeakyReLU, add, Flatten, Dense
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.utils import plot_model

class Network(object):
    def __init__(self, name, input_dim=[None, 6, 7], output_dim=7, lr=):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = 0.00025
        self.batch_size = 64
        self.epochs = 10
        self.build_model()
        self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        if not self.load_model():
            self.sess.run(tf.variables_initializer(self.graph.get_collection('variables')))

       
    def build_model(self):
        '''
        Builds the model
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.is_training = tf.placeholder(tf.bool, name="is_training")
            self.input_boards = tf.placeholder(tf.float32, shape=input_dim)
            x = tf.reshape(self.input_boards, [-1, input_dim[1], input_dim[2], 1])                

            # add a covolutional layer
            block = conv_bn_leaky_relu_layer(x, 42, (4,4))

            # add 6 res nets 
            for _ in range(6):
                 block = self.build_res_net(block, 42, (4,4))
            
            # Split NN into two  outputs
            self.value = self.build_value_output(block)
            self.policy = self.build_policy_output(block)

            # Loss Function
            self.target_pis = tf.placeholder(tf.float32, shape=[None, self.output_dim])
            self.target_vs = tf.placeholder(tf.float32, shape=[None])
            self.loss_pi = tf.losses.softmax_cross_entropy(self.target_pis, self.pi)
            self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1, ]))
            self.total_loss = self.loss_pi + self.loss_v
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss)
        

    def conv_bn_leaky_relu_layer(self, input_layer, filters, kernel_size):
        '''
        A helper function to createa a convolutional layer with batch normalization and leaky relu
        :param input_layer: 4D tensor
        :filter_shape: list. [height, width, in_channel, out_channel]
        :param stride: stride size for conv
        :return: 4D tensor. Y = LeakyRelu(BatchNormalization(conv(X)))
        '''
        conv_layer = tf.layers.conv2d(input_layer, filters=filters, kernel_size=kernel_size, 
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer, padding='SAME')
        bn_layer = tf.layers.batch_normalization(conv_layer, axis=3, training=self.is_training)
        output = tf.nn.leaky_relu(bn_layer)

        return output

    def residual_block(self, input_layer, filters=42, kernel_size=(4,4), use_bias=True):
        '''
        A helper function to create a ResNet block
        :param input_layer: 4D tensor
        :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
        :param first_block: if this is the first residual block of the whole network
        :return: 4D tensor.
        '''
        conv1_layer = self.conv_bn_leaky_relu_layer(input_layer, filters, kernel_size)
        conv2_layer = tf.layers.conv2d(conv1_layer, filters=filters, kernel_size=kernel_size,
                                         use_bias=use_bias, kernel_regularizer=tf.contrib.layers.l2_regularizer, padding='SAME')
        bn_layer = tf.layers.batch_normalization(conv2_layer, axis=3, training=self.is_training)
        res = input_layer + bn_layer
        output = tf.nn.leaky_relu(res)
        return output

    def build_value_output(self, input_layer):
        '''
        A helper function to create the value output
        :param input_layer: 4D tensor
        :return: value that represents the net confidence in winning the match
        '''
        conv_layer = self.conv_bn_leaky_relu_layer(input_layer, 4, (1,1), False)
        flat = tf.contrib.layers.flatten(conv_layer)
        fully_con_layer = tf.layers.dense(flat, units=20, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer)
        fully_con_layer = tf.nn.leaky_relu(fully_con_layer)
        output = tf.layers.dense(fully_con_layer, units=1, use_bias=False, activation='tanh', 
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer, name='value')
        return output

    def build_policy_output(self, input_layer):
        '''
        A helper function to create the policy output
        :param input_layer: 4D tensor
        :return: 1D tensor with probabilities of available moves
        '''
        conv_layer = self.conv_bn_leaky_relu_layer(input_layer, 2, (1,1), False)
        flat = tf.contrib.layers.flatten(conv_layer)
        output = tf.layers.dense(flat, units=self.output_dim, use_bias=False, activation='softmax', 
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer, name='policy')
        return output

    def load_model(self, folder='model', filename='model.pth.tar'):
        '''
        Loads pre trained weights 
        :param folder: name of the folder where model is stored
        :param filename: name of the pretrained model
        :return: True if model was loaded succesfully, otherwise false
        '''
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print("No model in path {}".format(filepath))
            return False

        self.saver.restore(self.sess, filepath)
        return True
    
    def save_model(self, folder='model', filename='model.pth.tar'):
        '''
        Loads pre trained weights 
        :param folder: name of the folder where model is stored
        :param filename: name of the pretrained model
        :return: True if model was loaded succesfully, otherwise false
        '''
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)

        if self.saver == None:
            self.saver = tf.train.Saver(self.nnet.graph.get_collection('variables'))
        with self.nnet.graph.as_default():
            self.saver.save(self.sess, filepath)

    def train(self, examples):
         '''
        Trains model with the examples provided
        :param examples: training examples in the form (board, pi, v)
        '''s
        pi_losses = []
        v_losses = []

        for epoch in range(self.epochs):
            print('EPOCH: ' + str(epoch + 1))

            batch_idx = 0

            for i in range(int(len(examples) / args.batch_size)):
                sample_ids = np.random.randint(len(examples), size=self.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                # predict and compute gradient and do SGD step
                input_dict = {self.input_boards: boards, self.target_pis: pis, self.target_vs: vs, self.is_training: True}

                # record loss
                self.sess.run(self.train_step, feed_dict=input_dict)
                pi_loss, v_loss = self.sess.run([self.loss_pi, self.loss_v], feed_dict=input_dict)
                pi_losses.append(pi_loss)
                v_losses.append(v_loss)
        
        self.plot_losses(pi_losses, v_losses)

    def plot_losses(self, pi_losses, v_losses):
        pass

    def predict(self, board):
         '''
        Trains model with the examples provided
        :param board: np array with board
        '''
        # Reshape board  input
        board = board[np.newaxis, :, :]
        prob, v = self.sess.run([self.policy, self.value], feed_dict={self.input_boards: board, self.is_training: False})

        return prob[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        with self.graph.as_default():
            self.saver.save(self.sess, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath + '.meta'):
            raise("No model in path {}".format(filepath))
        with self.nnet.graph.as_default():
            self.saver = tf.train.Saver()
        self.saver.restore(self.sess, filepath)