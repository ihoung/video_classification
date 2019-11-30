import tensorflow as tf

from abc import ABCMeta
from abc import abstractmethod
from nets import resnet_v2
from nets import inception
import preprocessing

slim = tf.contrib.slim

class BaseModel(object):
    """Abstract base class for any model."""
    __metaclass__ = ABCMeta

    def __init__(self, num_classes):
        """Constructor.

        Args:
            num_classes: Number of classes.
        """
        self._num_classes = num_classes

    @property
    def num_classes(self):
        return self._num_classes

    @abstractmethod
    def preprocess(self, inputs):
        """Input preprocessing. To be override by implementations.

        Args:
            inputs: A float32 tensor with shape [batch_size, height, width,
                num_channels] representing a batch of images.

        Returns:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, widht, num_channels] representing a batch of images.
        """
        pass

    @abstractmethod
    def predict(self, preprocessed_inputs):
        """Predict prediction tensors from inputs tensor.

        Outputs of this function can be passed to loss or postprocess functions.

        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.

        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        pass

    @abstractmethod
    def postprocess(self, prediction_dict):
        """Convert predicted output tensors to final forms.

        Args:
            prediction_dict: A dictionary holding prediction tensors.
            **params: Additional keyword arguments for specific implementations
                of specified models.

        Returns:
            A dictionary containing the postprocessed results.
        """
        pass

    @abstractmethod
    def loss(self, prediction_dict, groundtruth_lists):
        """Compute scalar loss tensors with respect to provided groundtruth.

        Args:
            prediction_dict: A dictionary holding prediction tensors.
            groundtruth_lists: A list of tensors holding groundtruth
                information, with one entry for each image in the batch.

        Returns:
            A dictionary mapping strings (loss names) to scalar tensors
                representing loss values.
        """
        pass


class ResNet152Model(BaseModel):
    """xxx definition."""

    def __init__(self, is_training, num_classes, default_image_size=256, fixed_resize_side=256):
        """Constructor.

        Args:
            is_training: A boolean indicating whether the training version of
                computation graph should be constructed.
            num_classes: Number of classes.
        """
        super(ResNet152Model, self).__init__(num_classes=num_classes)

        self._is_training = is_training
        self._default_image_size = default_image_size
        self._fixed_resize_side = fixed_resize_side

    def preprocess(self, inputs):
        """Predict prediction tensors from inputs tensor.

        Outputs of this function can be passed to loss or postprocess functions.

        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.

        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        preprocessed_inputs = preprocessing.preprocess_images(
            inputs, self._default_image_size, self._default_image_size,
            resize_side_min=self._fixed_resize_side,
            is_training=self._is_training,
            border_expand=True, normalize=False,
            preserving_aspect_ratio_resize=False)
        preprocessed_inputs = tf.cast(preprocessed_inputs, tf.float32)
        return preprocessed_inputs

    def predict(self, preprocessed_inputs):
        """Predict prediction tensors from inputs tensor.

        Outputs of this function can be passed to loss or postprocess functions.

        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.

        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        with slim.arg_scope(resnet_v2.resnet_arg_scope):
            net, endpoints = resnet_v2.resnet_v2_152(
                preprocessed_inputs, num_classes=None,
                is_training=self._is_training)
        net = tf.squeeze(net, axis=[1, 2])
        logits = slim.fully_connected(net, num_outputs=self.num_classes,
                                      activation_fn=None, scope='Predict')
        prediction_dict = {'logits': logits}
        return prediction_dict

    def postprocess(self, prediction_dict):
        """Convert predicted output tensors to final forms.

        Args:
            prediction_dict: A dictionary holding prediction tensors.
            **params: Additional keyword arguments for specific implementations
                of specified models.

        Returns:
            A dictionary containing the postprocessed results.
        """
        logits = prediction_dict['logits']
        logits = tf.nn.softmax(logits)
        classes = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)
        postprocessed_dict = {'classes': classes}
        return postprocessed_dict

    def loss(self, prediction_dict, groundtruth_lists):
        """Compute scalar loss tensors with respect to provided groundtruth.

        Args:
            prediction_dict: A dictionary holding prediction tensors.
            groundtruth_lists_dict: A dict of tensors holding groundtruth
                information, with one entry for each image in the batch.

        Returns:
            A dictionary mapping strings (loss names) to scalar tensors
                representing loss values.
        """
        logits = prediction_dict['logits']
        slim.losses.sparse_softmax_cross_entropy(
            logits=logits,
            labels=groundtruth_lists,
            scope='Loss')
        loss = slim.losses.get_total_loss()
        loss_dict = {'loss': loss}
        return loss_dict

    def accuracy(self, postprocessed_dict, groundtruth_lists):
        """Calculate accuracy.

        Args:
            postprocessed_dict: A dictionary containing the postprocessed
                results
            groundtruth_lists: A dict of tensors holding groundtruth
                information, with one entry for each image in the batch.

        Returns:
            accuracy: The scalar accuracy.
        """
        classes = postprocessed_dict['classes']
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(classes, groundtruth_lists), dtype=tf.float32))
        return accuracy


class InceptionResNetModel(BaseModel):
    """xxx definition."""

    def __init__(self, is_training, num_classes, default_image_size=256, fixed_resize_side=256):
        """Constructor.

        Args:
            is_training: A boolean indicating whether the training version of
                computation graph should be constructed.
            num_classes: Number of classes.
        """
        super(InceptionResNetModel, self).__init__(num_classes=num_classes)

        self._is_training = is_training
        self._default_image_size = default_image_size
        self._fixed_resize_side = fixed_resize_side

    def preprocess(self, inputs):
        """Predict prediction tensors from inputs tensor.

        Outputs of this function can be passed to loss or postprocess functions.

        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.

        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        preprocessed_inputs = preprocessing.preprocess_images(
            inputs, self._default_image_size, self._default_image_size,
            resize_side_min=self._fixed_resize_side,
            is_training=self._is_training,
            border_expand=True, normalize=False,
            preserving_aspect_ratio_resize=False)
        preprocessed_inputs = tf.cast(preprocessed_inputs, tf.float32)
        return preprocessed_inputs

    def predict(self, preprocessed_inputs):
        """Predict prediction tensors from inputs tensor.

        Outputs of this function can be passed to loss or postprocess functions.

        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.

        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
            net, endpoints = inception.inception_resnet_v2(
                preprocessed_inputs, num_classes=None,
                is_training=self._is_training)
        net = tf.squeeze(net, axis=[1, 2])
        logits = slim.fully_connected(net, num_outputs=self.num_classes,
                                      activation_fn=None, scope='Predict')
        prediction_dict = {'logits': logits}
        return prediction_dict

    def postprocess(self, prediction_dict):
        """Convert predicted output tensors to final forms.

        Args:
            prediction_dict: A dictionary holding prediction tensors.
            **params: Additional keyword arguments for specific implementations
                of specified models.

        Returns:
            A dictionary containing the postprocessed results.
        """
        logits = prediction_dict['logits']
        logits = tf.nn.softmax(logits)
        classes = tf.argmax(logits, axis=1)
        postprocessed_dict = {'classes': classes}
        return postprocessed_dict

    def loss(self, prediction_dict, groundtruth_lists):
        """Compute scalar loss tensors with respect to provided groundtruth.

        Args:
            prediction_dict: A dictionary holding prediction tensors.
            groundtruth_lists_dict: A dict of tensors holding groundtruth
                information, with one entry for each image in the batch.

        Returns:
            A dictionary mapping strings (loss names) to scalar tensors
                representing loss values.
        """
        logits = prediction_dict['logits']
        slim.losses.sparse_softmax_cross_entropy(
            logits=logits,
            labels=groundtruth_lists,
            scope='Loss')
        loss = slim.losses.get_total_loss()
        loss_dict = {'loss': loss}
        return loss_dict

    def accuracy(self, postprocessed_dict, groundtruth_lists):
        """Calculate accuracy.

        Args:
            postprocessed_dict: A dictionary containing the postprocessed
                results
            groundtruth_lists: A dict of tensors holding groundtruth
                information, with one entry for each image in the batch.

        Returns:
            accuracy: The scalar accuracy.
        """
        classes = postprocessed_dict['classes']
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(classes, groundtruth_lists), dtype=tf.float32))
        return accuracy
