from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten, GaussianNoise, Concatenate
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.initializers import GlorotNormal

import numpy as np
import tensorflow_addons as tfa

def scale_fn(x):
    return 1/(1.4**(x-1))


CLR_SCHEDULE_DEFAULT = tfa.optimizers.CyclicalLearningRate(
    initial_learning_rate=1e-6,
    maximal_learning_rate=0.02,
    scale_fn=scale_fn,
    step_size=50
)



def get_model(input_shape, n_classes):
    initializer = GlorotNormal()
    sigma = 0.01
    dropout = 0.5
    
    input_layer = Input(shape=input_shape)
    x = GaussianNoise(sigma)(input_layer)
    
    x = Dense(32, kernel_initializer=initializer)(x)
    x = GaussianNoise(sigma)(x)
    x = Dropout(dropout)(x)
    
    x = Dense(32, kernel_initializer=initializer)(x)
    x = GaussianNoise(sigma)(x)
    x = Dropout(dropout)(x)
    

    output = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output)

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(CLR_SCHEDULE_DEFAULT),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    return model