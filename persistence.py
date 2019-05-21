import tensorflow as tf

def serving_input_receiver_fn():
  feature_spec = {
    'ax': tf.FixedLenFeature([], dtype=tf.float32),
    'ay': tf.FixedLenFeature([], dtype=tf.float32),
    'az': tf.FixedLenFeature([], dtype=tf.float32),
    'gx': tf.FixedLenFeature([], dtype=tf.float32),
    'gy': tf.FixedLenFeature([], dtype=tf.float32),
    'gz': tf.FixedLenFeature([], dtype=tf.float32),
  }
  serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None], name='input_tensors')

  receiver_tensors = { 'classifier_inputs': serialized_tf_example }

  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)