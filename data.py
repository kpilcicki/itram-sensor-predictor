import tensorflow as tf
from constants import used_columns

def input_fn(all_data, num_epochs, batch_size, shuffle):
  for col in all_data.columns:
    if not col in used_columns:
      all_data.drop(col, axis=1, inplace=True)
  
  label = all_data['imInTram']

  ds = tf.data.Dataset.from_tensor_slices((dict(all_data), label))

  if shuffle:
    ds = ds.shuffle(10000)

  ds = ds.batch(batch_size).repeat(num_epochs)
  return ds

def split_data_frame(df, train_set_fraction):
  DATA_SIZE = df.count()['ax']
  train_size = int(DATA_SIZE * 0.7)
  test_size = DATA_SIZE - train_size

  df = df.sample(frac=1).reset_index(drop=True)

  train_df = df.iloc[:train_size].reset_index(drop=True)
  test_df = df.iloc[train_size:].reset_index(drop=True)

  return (train_df, test_df)
