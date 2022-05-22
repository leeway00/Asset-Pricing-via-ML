import numpy as np
import tensorflow as tf
from MLutils.BaseUtils import *
from datetime import datetime

class WindowGenerator():
  def __init__(self, input_width, label_width=1, shift = 0,
               train_df=None, val_df=None, test_df=None,
               label_columns=None, batch_size = 32, dtype = np.float32):
    # Store the raw data.
    self.data = {
      'train' : train_df,
      'val' : val_df,
      'test' : test_df,
    }
    self.date = {}
    
    if 'Date' in train_df.columns:
      for key in ['train', 'val', 'test']:
        self.date[key] = self.data[key]['Date'].values
        self.data[key].drop(['Date'], axis=1, inplace= True)

    self.dtype = dtype
    self.batch_size = batch_size
    
    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_dim = test_df.values.shape[1]-len(label_columns)
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])


  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack(
          [labels[:, :, self.column_indices[name]] for name in self.label_columns],
          axis=-1)
      inputs = tf.gather(inputs , [i[1] for i in self.column_indices.items() if i[0] not in self.label_columns], axis = -1)
    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels
  
  
  def make_dataset(self, data):
    data = np.array(data, dtype=self.dtype)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle = False,
        batch_size=self.batch_size,)

    ds = ds.map(self.split_window)

    return ds

  @property
  def train(self):
    return self.make_dataset(self.data["train"])

  @property
  def val(self):
    return self.make_dataset(self.data["val"])

  @property
  def test(self):
    return self.make_dataset(self.data["test"])

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result
  
  def y_true(self):
    return tf.cast(tf.concat([tf.reshape(i[1], [-1,1]) for  i in self.test], axis=0)>0, self.dtype)

def compile_and_fit(model, window, patience=10, epochs=100, tensorboard= True, 
                    ):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
  
  log_dir = "logs/fit/" + datetime.now().strftime("%Y/%m/%d-%H:%M:%S")
  if tensorboard:
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [early_stopping, tensorboard_callback]
  else:
    callbacks = [early_stopping]

  
  history = model.fit(window.train, epochs=epochs,
                      validation_data=window.val,
                      callbacks=callbacks)
  return history