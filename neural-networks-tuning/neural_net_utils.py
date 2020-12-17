import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics


def keras_compile(model):
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.BinaryCrossentropy(),
        metrics=[metrics.BinaryAccuracy()],
    )


def to_tensor(x_train, x_test, y_train, y_test, x_type=tf.float32):
    return tf.constant(x_train, dtype=x_type), \
           tf.constant(x_test, dtype=x_type), \
           tf.constant(y_train, dtype=tf.int8), \
           tf.constant(y_test, dtype=tf.int8)
