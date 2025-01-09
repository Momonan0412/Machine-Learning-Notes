import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, LSTM, Input
from keras.api.optimizers import Adam
from keras import regularizers
# https://stackoverflow.com/questions/63006575/what-is-the-difference-between-maxpool-and-maxpooling-layers-in-kera
class ModelBuilder:
    """
    input_shape: The shape of the input data (e.g., (128, 128, 3) for an image).
    num_classification: The number of output classes (for classification).
    model_type: Whether to build a 'sequential' model or a 'functional' model.
    **kwargs: Any additional arguments for flexibility (like layer configurations, optimizer settings, etc.).
    """
    def __init__(self, input_shape, num_classification, model_type="sequential", **kwargs):
        self._input_shape = input_shape
        self._num_classification = num_classification
        self._model_type = model_type
        self._kwargs = kwargs
    
    def _build_model(self, **kwargs):
        print("Debug")
        if self._model_type == "sequential":
            return self._build_sequential_model(kwargs.get('learning_rate'))

    def _build_sequential_model(self, learning_rate):
        model = Sequential()
        model.add(Input(shape=self._input_shape)) # Define the input shape for the first layer
        # Best Practice
        # https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
        for layer_config in self._kwargs.get('layers', []):
            if layer_config['type'] == 'conv':
                model.add(Conv2D(filters=layer_config['filters'],
                                 kernel_size=layer_config['kernel_size'],
                                 activation=layer_config['activation']))
            if layer_config['type'] == 'maxpool':
                model.add(MaxPooling2D(pool_size=layer_config['pool_size'], strides=layer_config['strides'], padding=layer_config['padding']))
            if layer_config['type'] == 'batch_normalization':
                model.add(BatchNormalization())
            if layer_config['type'] == 'flatten':
                model.add(Flatten())
            if layer_config['type'] == 'dense':
                model.add(Dense(units=layer_config['units'],
                                activation=layer_config['activation']))
            if layer_config['type'] == 'dropout':
                model.add(Dropout(rate=layer_config['rate']))
            if layer_config['type'] == 'lstm':
                model.add(LSTM(units=layer_config['units'], return_sequences=layer_config['return_sequences']))
                
                # model.add(LSTM(
                #     units=layer_config['units'],
                #     return_sequences=layer_config['return_sequences'],
                #     activation='tanh', recurrent_activation='sigmoid'))  # Equivalent to the default
                
        model.add(Dense(self._num_classification, activation='softmax'))
        # Compile the model with default settings
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
        return model