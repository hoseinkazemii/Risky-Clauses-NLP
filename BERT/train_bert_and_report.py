import numpy as np

from Preprocessing import load_data
from Preprocessing import split_data
from Preprocessing import load_cleaned_X
from Preprocessing import split_val_test
from MLModels import train_test_split_

from ._tokenize_bert import tokenize_bert
from ._data_loader import data_loader
from ._sequence_to_tensor import sequence_to_tensor
from ._initialize_bert import initialize_bert
from ._fine_tune import fine_tune
from ._start_training import start_training

from ._predict_test import predict_test

# def to_categorical(y, num_classes):
#     """ 1-hot encodes a tensor """
#     return np.eye(num_classes, dtype='uint8')[y]

def train_bert_and_report(**params):

	df = load_data(**params)
	_, Y = split_data(df, **params)
	X = load_cleaned_X(**params)

	X_train, X_test, Y_train, Y_test = train_test_split_(X, Y, **params)
	X_val, X_test, Y_val, Y_test = split_val_test(X_test, Y_test, **params)

	X_train = tokenize_bert(X_train, **params)
	X_val = tokenize_bert(X_val, **params)
	X_test = tokenize_bert(X_test, **params)

	X_train_seq, X_train_mask, Y_train = sequence_to_tensor(X_train, Y_train, **params)
	X_val_seq, X_val_mask, Y_val = sequence_to_tensor(X_val, Y_val, **params)
	X_test_seq, X_test_mask, Y_test = sequence_to_tensor(X_test, Y_test, **params)
	train_dataloader, val_dataloader = data_loader(X_train_seq, X_train_mask, Y_train,
			    								   X_val_seq, X_val_mask, Y_val, **params)
	
	bert = initialize_bert(X_train_seq, X_train_mask, **params)
	model, optimizer, weights, cross_entropy, device = fine_tune(bert, Y_train, **params)
	start_training(model, train_dataloader, val_dataloader, cross_entropy, optimizer, **params)
	predict_test(model, X_test_seq, X_test_mask, Y_test, **params)