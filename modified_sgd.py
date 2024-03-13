# Modified_SGD
import tensorflow as tf

class Modified_SGD(tf.keras.optimizers.SGD):
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False, lr_multipliers=None, momentum_multipliers=None, name="Modified_SGD", **kwargs):
        # Remove decay from kwargs
        kwargs.pop('decay', None)

        super(Modified_SGD, self).__init__(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov, name=name, **kwargs)
        self.lr_multipliers = lr_multipliers
        self.momentum_multipliers = momentum_multipliers

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        momentum_var = self.get_slot(var, "momentum")

        if self.nesterov:
            update = grad * lr_t + self.momentum * momentum_var
            var_update = var.assign_sub(update)
        else:
            var_update = var.assign_sub(lr_t * grad)

        if self.nesterov:
            momentum_update = momentum_var.assign_add(self.momentum * momentum_var + grad * lr_t)
            return tf.group([var_update, momentum_update])
        else:
            return var_update

    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        momentum_var = self.get_slot(var, "momentum")

        if self.nesterov:
            update = grad.values * lr_t + self.momentum * momentum_var
            var_update = var.scatter_sub(indices, update)
        else:
            var_update = var.scatter_sub(indices, lr_t * grad.values)

        if self.nesterov:
            momentum_update = momentum_var.scatter_add(indices, self.momentum * momentum_var + grad.values * lr_t)
            return tf.group([var_update, momentum_update])
        else:
            return var_update

    def _decayed_lr(self, var_dtype):
        lr_t = self.learning_rate
        if not isinstance(lr_t, (tf.Tensor, tf.Variable)):
            initial_lr_t = tf.convert_to_tensor(lr_t, dtype=var_dtype)
            lr_t = tf.compat.v1.train.polynomial_decay(
                initial_lr_t,
                self.iterations,
                self.decay_steps,
                end_learning_rate=self.end_learning_rate,
                power=1.0,
                cycle=False)
        return lr_t.numpy() if hasattr(lr_t, 'numpy') else lr_t

    def set_lr_schedule(self, decay_steps, end_learning_rate):
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
