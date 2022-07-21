import os
import numpy as np

from multiprocessing import current_process
if current_process().name == "MainProcess":
	from tensorflow.keras.optimizers import Adam
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.models import load_model
	from tensorflow.keras.layers import Embedding, LSTM, Dense, GRU
	from tensorflow.keras.layers import Flatten, Dropout
	from tensorflow.keras.regularizers import l1, l2
	from tensorflow.keras import backend as K
	from tensorflow.math import argmax
	import tensorflow as tf



def construct_network(embedding_weights, vocab_size,**params):
	lr = params.get("lr")
	pretrained_emb = params.get("pretrained_emb")
	emb_dimension = params.get("emb_dimension")
	maxlen = params.get("maxlen")
	dropout = params.get("dropout")
	rec_dropout = params.get("rec_dropout")
	LSTM_cells = params.get("LSTM_cells")
	should_warm_up = params.get("should_warm_up")
	embedding_type = params.get("embedding_type")
	network_type = params.get("network_type")
	layers = params.get("layers")
	dense_layers = params.get("dense_layers")

	if should_warm_up:
		direc = f"./MLModels/TheModel-{embedding_type}.h5"
		print ("\n\n-----Model is loaded-----\n\n")
		if os.path.exists(direc):
			model = load_model(direc)
			return model


	print('constructing the deep neural network network...')

	if pretrained_emb:
		model = Sequential()
		model.add(Embedding(vocab_size, emb_dimension, input_length = maxlen,
		                weights = [embedding_weights], trainable = False))

		if network_type == "LSTM":

			for k, l in enumerate(layers):
				model.add(LSTM(l, dropout = dropout,
									recurrent_dropout = rec_dropout,
									return_sequences = True, name=f"{network_type}_{k}"))
		elif network_type == "GRU":

			for k, l in enumerate(layers):
				model.add(GRU(l, dropout = dropout,
									recurrent_dropout = rec_dropout,
									return_sequences = True, name=f"{network_type}_{k}"))

		else:
			raise ValueError("network_type should be one of 'LSTM', 'GRU' or 'CNN'")


		for k, l in enumerate(dense_layers):
			model.add(Dense(l, activation = 'relu', name=f"Dense_{k}"))
			model.add(Dropout(dropout))

		model.add(Flatten())
		model.add(Dense(1, activation = 'sigmoid'))
		# model.summary()

	else:
		model = Sequential()
		model.add(Embedding(vocab_size, emb_dimension, input_length = maxlen))

		for k, l in enumerate(layers):
			model.add(network_type(l, dropout = dropout,
								recurrent_dropout = rec_dropout,
								return_sequences = True, name=f"{network_type}_{k}"))

		for k, l in enumerate(dense_layers):
			model.add(Dense(l, activation = 'relu', name=f"Dense_{k}"))
			model.add(Dropout(dropout))

		model.add(Flatten())
		model.add(Dense(1, activation = 'sigmoid'))		
		# model.summary()

	opt = Adam(learning_rate = lr)
	model.compile(optimizer = opt,
					loss = 'binary_crossentropy',
					metrics = ['accuracy'])

	return model


def recall_m(y_true, y_pred):
	y_true = tf.gather(y_true, 1, axis = 1)
	y_pred = tf.gather(y_pred, 1, axis = 1)

	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def precision_m(y_true, y_pred):
	y_true = tf.gather(y_true, 1, axis = 1)
	y_pred = tf.gather(y_pred, 1, axis = 1)

	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def f1_m(y_true, y_pred):
	precision = precision_m(y_true, y_pred)
	recall = recall_m(y_true, y_pred)
	return 2*((precision*recall)/(precision+recall+K.epsilon()))