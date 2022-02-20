import numpy as np
from keras import backend as k
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers
from keras.utils.generic_utils import get_custom_objects


def _moments(x, axes, shift=None, keep_dims=False):
    """
    Wrapper over tensorflow backend call
    """
    if k.backend() == "tensorflow":
        import tensorflow as tf
        return tf.nn.moments(x, axes, shift=shift, keep_dims=keep_dims)
    elif k.backend() == "theano":
        import theano.tensor as t
        mean_batch = t.mean(x, axis=axes, keepdims=keep_dims)
        var_batch = t.var(x, axis=axes, keepdims=keep_dims)
        return mean_batch, var_batch
    else:
        raise RuntimeError("Currently does not support CNTK backend")


class BatchRenormalization(Layer):
    """
    Batch renormalization layer (Sergey Ioffe, 2017).
    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        epsilon: small float > 0. Fuzz parameter.
            Theano expects epsilon >= 1e-5.
        mode: integer, 0, 1 or 2.
            - 0: feature-wise normalization.
                Each feature map in the input will
                be normalized separately. The axis on which
                to normalize is specified by the `axis` argument.
                Note that if the input is a 4D image tensor
                using Theano conventions (samples, channels, rows, cols)
                then you should set `axis` to `1` to normalize along
                the channels axis.
                During training and testing we use running averages
                computed during the training phase to normalize the data
            - 1: sample-wise normalization. This mode assumes a 2D input.
            - 2: feature-wise normalization, like mode 0, but
                using per-batch statistics to normalize the data during both
                testing and training.
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        r_max_value: Upper limit of the value of r_max.
        d_max_value: Upper limit of the value of d_max.
        t_delta: At each iteration, increment the value of t by t_delta.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
            Note that the order of this list is [gamma, beta, mean, std]
        beta_init: name of initialization function for shift parameter
            (see [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don"t pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don"t pass a `weights` argument.
        gamma_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the gamma vector.
        beta_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the beta vector.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models]
        - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift]
    """
    def __init__(self, epsilon=1e-3, mode=0, axis=-1, momentum=0.99,
                 r_max_value=3., d_max_value=5., t_delta=1e-3, weights=None, beta_init="zero",
                 gamma_init="one", gamma_regularizer=None, beta_regularizer=None,
                 **kwargs):
        self.supports_masking = True
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.epsilon = epsilon
        self.mode = mode
        self.axis = axis
        self.momentum = momentum
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.initial_weights = weights
        self.r_max_value = r_max_value
        self.d_max_value = d_max_value
        self.t_delta = t_delta
        if self.mode == 0:
            self.uses_learning_phase = True
        super(BatchRenormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)

        self.gamma = self.add_weight(shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name="{}_gamma".format(self.name))
        self.beta = self.add_weight(shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name="{}_beta".format(self.name))
        self.running_mean = self.add_weight(shape, initializer="zero",
                                            name="{}_running_mean".format(self.name),
                                            trainable=False)
        # Note: running_std actually holds the running variance, not the running std.
        self.running_std = self.add_weight(shape, initializer="one",
                                           name="{}_running_std".format(self.name),
                                           trainable=False)

        self.r_max = k.variable(np.ones((1,)), name="{}_r_max".format(self.name))

        self.d_max = k.variable(np.zeros((1,)), name="{}_d_max".format(self.name))

        self.t = k.variable(np.zeros((1,)), name="{}_t".format(self.name))

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x, mask=None):
        if self.mode == 0 or self.mode == 2:
            assert self.built, "Layer must be built before being called"
            input_shape = k.int_shape(x)

            reduction_axes = list(range(len(input_shape)))
            del reduction_axes[self.axis]
            broadcast_shape = [1] * len(input_shape)
            broadcast_shape[self.axis] = input_shape[self.axis]

            mean_batch, var_batch = _moments(x, reduction_axes, shift=None, keep_dims=False)
            std_batch = (k.sqrt(var_batch + self.epsilon))

            r_max_value = k.get_value(self.r_max)
            r = std_batch / (k.sqrt(self.running_std + self.epsilon))
            r = k.stop_gradient(k.clip(r, 1 / r_max_value, r_max_value))

            d_max_value = k.get_value(self.d_max)
            d = (mean_batch - self.running_mean) / k.sqrt(self.running_std + self.epsilon)
            d = k.stop_gradient(k.clip(d, -d_max_value, d_max_value))

            if sorted(reduction_axes) == range(k.ndim(x))[:-1]:
                x_normed_batch = (x - mean_batch) / std_batch
                x_normed = (x_normed_batch * r + d) * self.gamma + self.beta
            else:
                # need broadcasting
                broadcast_mean = k.reshape(mean_batch, broadcast_shape)
                broadcast_std = k.reshape(std_batch, broadcast_shape)
                broadcast_r = k.reshape(r, broadcast_shape)
                broadcast_d = k.reshape(d, broadcast_shape)
                broadcast_beta = k.reshape(self.beta, broadcast_shape)
                broadcast_gamma = k.reshape(self.gamma, broadcast_shape)

                x_normed_batch = (x - broadcast_mean) / broadcast_std
                x_normed = (x_normed_batch * broadcast_r + broadcast_d) * broadcast_gamma + broadcast_beta

            # explicit update to moving mean and standard deviation
            self.add_update([k.moving_average_update(self.running_mean, mean_batch, self.momentum),
                             k.moving_average_update(self.running_std, std_batch ** 2, self.momentum)], x)

            # update r_max and d_max
            r_val = self.r_max_value / (1 + (self.r_max_value - 1) * k.exp(-self.t))
            d_val = self.d_max_value / (1 + ((self.d_max_value / 1e-3) - 1) * k.exp(-(2 * self.t)))

            self.add_update([k.update(self.r_max, r_val),
                             k.update(self.d_max, d_val),
                             k.update_add(self.t, k.variable(np.array([self.t_delta])))], x)

            if self.mode == 0:
                if sorted(reduction_axes) == range(k.ndim(x))[:-1]:
                    x_normed_running = k.batch_normalization(
                        x, self.running_mean, self.running_std,
                        self.beta, self.gamma,
                        epsilon=self.epsilon)
                else:
                    # need broadcasting
                    broadcast_running_mean = k.reshape(self.running_mean, broadcast_shape)
                    broadcast_running_std = k.reshape(self.running_std, broadcast_shape)
                    broadcast_beta = k.reshape(self.beta, broadcast_shape)
                    broadcast_gamma = k.reshape(self.gamma, broadcast_shape)
                    x_normed_running = k.batch_normalization(
                        x, broadcast_running_mean, broadcast_running_std,
                        broadcast_beta, broadcast_gamma,
                        epsilon=self.epsilon)

                # pick the normalized form of x corresponding to the training phase
                # for batch renormalization, inference time remains same as batchnorm
                x_normed = k.in_train_phase(x_normed, x_normed_running)

        elif self.mode == 1:
            # sample-wise normalization
            m = k.mean(x, axis=self.axis, keepdims=True)
            std = k.sqrt(k.var(x, axis=self.axis, keepdims=True) + self.epsilon)
            x_normed_batch = (x - m) / (std + self.epsilon)

            r_max_value = k.get_value(self.r_max)
            r = std / (self.running_std + self.epsilon)
            r = k.stop_gradient(k.clip(r, 1 / r_max_value, r_max_value))

            d_max_value = k.get_value(self.d_max)
            d = (m - self.running_mean) / (self.running_std + self.epsilon)
            d = k.stop_gradient(k.clip(d, -d_max_value, d_max_value))

            x_normed = ((x_normed_batch * r) + d) * self.gamma + self.beta

            # update r_max and d_max
            t_val = k.get_value(self.t)
            r_val = self.r_max_value / (1 + (self.r_max_value - 1) * np.exp(-t_val))
            d_val = self.d_max_value / (1 + ((self.d_max_value / 1e-3) - 1) * np.exp(-(2 * t_val)))
            t_val += float(self.t_delta)

            self.add_update([k.update(self.r_max, r_val),
                             k.update(self.d_max, d_val),
                             k.update(self.t, t_val)], x)
        return x_normed

    def get_config(self):
        config = {"epsilon": self.epsilon,
                  "mode": self.mode,
                  "axis": self.axis,
                  "gamma_regularizer": regularizers.serialize(self.gamma_regularizer),
                  "beta_regularizer": regularizers.serialize(self.beta_regularizer),
                  "momentum": self.momentum,
                  "r_max_value": self.r_max_value,
                  "d_max_value": self.d_max_value,
                  "t_delta": self.t_delta}
        base_config = super(BatchRenormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({"BatchRenormalization": BatchRenormalization})
