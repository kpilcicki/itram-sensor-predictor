import pandas
import tensorflow as tf
import tensorflow.feature_column as fc
import os
import sys
import functools
from constants import used_columns
from persistence import serving_input_receiver_fn

from display import pprint_result, peek_classification
from data import input_fn, split_data_frame, prepare_data
import plots as pl

# tf.enable_eager_execution()

# readings_file = 'tram_sensor_readings.json'
readings_file = 'new-sensor-readings.json'

data_frame = pandas.read_json(readings_file)

data_frame = prepare_data(data_frame)


train_df, test_df = split_data_frame(data_frame, train_set_fraction=0.6)

ax = fc.numeric_column('ax')
ay = fc.numeric_column('ay')
az = fc.numeric_column('az')
gx = fc.numeric_column('gx')
gy = fc.numeric_column('gy')
gz = fc.numeric_column('gz')


big_hidden_units=[1024,512, 256]
ok_hidden_units=[15,100]
mid_hidden_units=[20, 40, 20]
small_hidden_units=[20, 1]
EPOCHS_NUM = 100
# EPOCHS_NUM = 1000
ADAM_LEARNING_RATE = 0.005
proxAdagrad_LEARNING_RATE = 0.15
ADADELTA_LEARNING_RATE = 0.01
DECAY_STEPS = 100

adam_optimizer = lambda: tf.train.AdamOptimizer(
      learning_rate=tf.train.exponential_decay(
        learning_rate=ADAM_LEARNING_RATE,
        global_step=tf.train.get_global_step(),
        decay_steps=DECAY_STEPS,
        decay_rate=0.96))
adagrad_optimizer = tf.train.ProximalAdagradOptimizer(
      learning_rate=proxAdagrad_LEARNING_RATE,
      l1_regularization_strength=0.001)

adadelta_optimizer = tf.train.AdadeltaOptimizer(
  learning_rate = ADADELTA_LEARNING_RATE
)



tested_nums = []
accuracies=[]
precisions=[]
recalls=[]
for learning_rate in [x * 0.001 for x in [1]]:
  print(f"testing: {learning_rate}")
  test_hidden_units = ok_hidden_units
  estimator = tf.estimator.DNNClassifier(
    feature_columns=[ax, ay, az, gx, gy, gz],
    hidden_units=test_hidden_units,
    optimizer=adadelta_optimizer
  )

  estimator.train(input_fn=lambda:input_fn(train_df, num_epochs=EPOCHS_NUM, shuffle=False, batch_size=64), steps=None)

  result = estimator.evaluate(lambda:input_fn(test_df, num_epochs=1, shuffle=False, batch_size=64))
  tested_nums.append(learning_rate)
  accuracies.append(result["accuracy"])
  precisions.append(result["precision"])
  recalls.append(result["recall"])
  print(f"A: {result['accuracy']},  B: {result['precision']}")
  estimator.export_savedmodel(export_dir_base='adam-dnn', serving_input_receiver_fn=serving_input_receiver_fn)


pl.show_multiple_series([accuracies, precisions, recalls])
pprint_result(result)
peek_classification(estimator, test_df)