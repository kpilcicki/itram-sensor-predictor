import pandas
import tensorflow as tf
import tensorflow.feature_column as fc
import os
import sys
import functools

from display import pprint_result, peek_classification
from data import input_fn, split_data_frame

tf.enable_eager_execution()

readings_file = 'tram_sensor_readings.json'
data_frame = pandas.read_json(readings_file)

train_df, test_df = split_data_frame(data_frame, train_set_fraction=0.7)

ax = fc.numeric_column('ax')
ay = fc.numeric_column('ay')
az = fc.numeric_column('az')
gx = fc.numeric_column('gx')
gy = fc.numeric_column('gy')
gz = fc.numeric_column('gz')

estimator = tf.estimator.LinearClassifier(
  feature_columns=[ax, ay, az, gx, gy, gz]
)

estimator.train(input_fn=lambda:input_fn(train_df, num_epochs=2, shuffle=True, batch_size=64), steps=5000)

result = estimator.evaluate(lambda:input_fn(test_df, num_epochs=1, shuffle=False, batch_size=64))

pprint_result(result)
peek_classification(estimator, test_df)
