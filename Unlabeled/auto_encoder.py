import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import slim


def encoder(inputs):
    with tf.variable_scope('Encoder'):
        conv1 = tf.layers.conv2d(inputs, 32, (3, 3),
                                 strides=(1, 1), padding='same',
                                 kernel_initializer=slim.xavier_initializer(),
                                 kernel_regularizer=layers.l2_regularizer(0.0005),
                                 bias_initializer=slim.xavier_initializer(),
                                 activation=tf.nn.relu,
                                 name='conv1')
        avgpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool1')
        conv2 = tf.layers.conv2d(avgpool1, 16, (3, 3),
                                 strides=(1, 1), padding='same',
                                 kernel_initializer=slim.xavier_initializer(),
                                 kernel_regularizer=layers.l2_regularizer(0.0005),
                                 bias_initializer=slim.xavier_initializer(),
                                 activation=tf.nn.relu,
                                 name='conv2')
        avgpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool2')
        conv3 = tf.layers.conv2d(avgpool2, 8, (3, 3),
                                 strides=(1, 1), padding='same',
                                 kernel_initializer=slim.xavier_initializer(),
                                 kernel_regularizer=layers.l2_regularizer(0.0005),
                                 bias_initializer=slim.xavier_initializer(),
                                 activation=tf.nn.relu,
                                 name='conv3')
        encoded = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool3')
        return encoded


def decoder(encoded):
    encoded_shape = encoded.get_shape().as_list()
    height, width = encoded_shape[1:3]
    with tf.variable_scope('Decoder'):
        upsample1 = tf.image.resize_images(encoded, size=(height*2, width*2),
                                           method=tf.image.ResizeMethod.BILINEAR)
        conv4 = tf.layers.conv2d(upsample1, 16, (3, 3),
                                            strides=(1, 1), padding='same',
                                            kernel_initializer=slim.xavier_initializer(),
                                            kernel_regularizer=layers.l2_regularizer(0.0005),
                                            bias_initializer=slim.xavier_initializer(),
                                            activation=tf.nn.relu,
                                            name='conv4')
        upsample2 = tf.image.resize_images(conv4, size=(height*4, width*4),
                                           method=tf.image.ResizeMethod.BILINEAR)
        conv5 = tf.layers.conv2d(upsample2, 32, (3, 3),
                                            strides=(1, 1), padding='same',
                                            kernel_initializer=slim.xavier_initializer(),
                                            kernel_regularizer=layers.l2_regularizer(0.0005),
                                            bias_initializer=slim.xavier_initializer(),
                                            activation=tf.nn.relu,
                                            name='conv5')
        upsample3 = tf.image.resize_images(conv5, size=(height*8, width*8),
                                           method=tf.image.ResizeMethod.BILINEAR)
        outputs = tf.layers.conv2d(upsample3, 3, (3, 3),
                                             strides=(1, 1), padding='same',
                                             kernel_initializer=slim.xavier_initializer(),
                                             kernel_regularizer=layers.l2_regularizer(0.0005),
                                             bias_initializer=slim.xavier_initializer(),
                                             activation=tf.nn.relu,
                                             name='outputs')
        return outputs


def loss(outputs, original_img):
    with tf.variable_scope('Loss'):
        _loss = tf.reduce_mean(tf.losses.mean_squared_error(original_img, outputs)) + \
               tf.reduce_mean(tf.losses.get_regularization_losses())
    return _loss


def accuracy():
    pass


if __name__ == '__main__':
    pass
