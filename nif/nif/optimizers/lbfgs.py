# code comes from: https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993
import numpy as np
import tensorflow as tf
from tensorflow_probability.python.optimizer import lbfgs_minimize


def function_factory(model, loss, train_x, train_y, display_epoch):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.

    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training demo.
        train_y [in]: the output part of training demo.

    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = []  # stitch indices
    part = []  # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        part.extend([i] * n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.

        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.

        This function is created by function_factory.

        Args:
           params_1d [in]: a 1D tf.Tensor.

        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            loss_value = loss(model(train_x, training=True), train_y)

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        f.iter.assign_add(1)

        if f.iter % display_epoch == 0:
            tf.print("Epoch:", f.iter, "loss:", loss_value)

        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[loss_value], Tout=[])

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f


class TFPLBFGS(object):
    def __init__(self, model, loss_fun, inps, outs, display_epoch=1):
        # demo + keras model -> function for l-bfgs
        self.func = function_factory(model, loss_fun, inps, outs, display_epoch)
        self.model = model

    def minimize(self, rounds=50, max_iter=50):
        for _ in range(rounds):
            results = lbfgs_minimize(
                value_and_gradients_function=self.func,
                initial_position=tf.dynamic_stitch(
                    self.func.idx, self.model.trainable_variables
                ),
                num_correction_pairs=20,
                tolerance=1e-15,
                x_tolerance=1e-15,
                f_relative_tolerance=1e-15,
                parallel_iterations=1,
                max_iterations=max_iter,
                max_line_search_iterations=100,
            )
            self.func.assign_new_model_parameters(results.position)
        # print("loss = %8.5f" % results.objective_value)

    @property
    def history(self):
        history = list(map(lambda x: x.numpy(), self.func.history))
        iteration = np.arange(1, len(history) + 1)
        return {"iteration": iteration, "loss": history}
