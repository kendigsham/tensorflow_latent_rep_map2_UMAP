#!/usr/bin/env python3

import scanpy as sc
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kerastuner import HyperModel
from kerastuner.tuners import Hyperband #,RandomSearch, BayesianOptimization,  
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout


print('loading in data')

adata = sc.read('dataA_dataB_17808_1000_scaled_minax_umap.h5ad')

latent_rep= adata.obsm['X_minmax_latent']
input_shape=latent_rep.shape[1]
diffusion_temp = adata.obsm['X_umap']

## dim X
print('dim_y')
diffusion_dim = diffusion_temp[:,1]


print('splitting data into train and test')

latent_train, latent_test, umap_train, umap_test = train_test_split(latent_rep, diffusion_dim, test_size=0.15, random_state=42, shuffle=True)

print('defining the model')

def build_model(hp):
	model = keras.models.Sequential()

	model.add(Dense(hp.Int('first_layer',
        min_value=32,
        max_value=256,
        step=32), 
		activation=hp.Choice('first_activation', ['sigmoid','relu'], default='relu'), 
		input_shape=[input_shape]))

	model.add(Dropout(hp.Float('first_dropout', 0, 0.5, step=0.1, default=0.2)))

	for i in range(hp.Int('n_layers', 2, 20)):
		model.add(Dense(hp.Int(f'layer_{i}',
                                min_value=32,
                                max_value=256,
                                step=32), 
			activation=hp.Choice(f'activation_{i}', ['sigmoid','relu'], default='relu')))
		model.add(Dropout(hp.Float(f'dropout_{i}', 0, 0.5, step=0.1, default=0.2)))
		# model.add(Dense(32, activation='sigmoid'))
		# model.add(Dropout(0.2))

	model.add(Dense(1))


	optimizer = tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-5, 1e-2, sampling='log',default=1e-3))
	# loss_object = tf.keras.losses.MeanSquaredError()

	model.compile(loss='mse',
	                optimizer=optimizer,
	                metrics=['mae','mse']) # dont use accuracy its for classification

	return model


# tuner = RandomSearch(
#     build_model,
#     objective='val_accuracy',
#     max_trials=20, # how many variations on model?
#     executions_per_trial=2, # how many trials per variation? (same model could perform differently)
#     directory='projection_umap_hypterparameter_search')

HYPERBAND_MAX_EPOCHS=600

print('making tuner object')

tuner = Hyperband(
    build_model,
    max_epochs=HYPERBAND_MAX_EPOCHS,
    objective='val_mse',
    executions_per_trial=2, # how many trials per variation? (same model could perform differently)
    directory='projection_umap_Hyperband_dim_y',
    project_name='dataA_dataB')

print(tuner.search_space_summary())


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

print('starting the actual hyperparameter search')

tuner.search(x=latent_train,
             y=umap_train,
             epochs=HYPERBAND_MAX_EPOCHS,
             verbose=2, # erbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
             shuffle=True,
             batch_size=64,
             validation_split=0.15,
             callbacks=[early_stop])

print(tuner.results_summary())


##################
# 
#################

print('getting the best model first')

best_model = tuner.get_best_models(num_models=1)[0]

best_model.save('best_model_projection_umap_dim_y')


print('\n# Evaluate on test data')
results = best_model.evaluate(latent_test, umap_test, batch_size=32)
print(best_model.metrics_names)
print(results)

print('saving the tuner object')

with open("tuner_projection_umap_dim_y.pkl", "wb") as f:
    pickle.dump(tuner, f)

print('all done !!!!!!!!')
