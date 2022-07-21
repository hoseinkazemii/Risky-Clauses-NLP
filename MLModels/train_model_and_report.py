import pandas as pd

from embedding import load_network_info
from Preprocessing import load_XYs

from ._construct_network import construct_network
from ._train_model import train_model
from ._evaluate_classification import evaluate_classification
from ._get_callbacks import get_callbacks


def train_model_and_report(**params):

	index_dict, vocab_size, embedding_weights = load_network_info(**params)

	embedding_type = params.get("embedding_type")
	model = construct_network(embedding_weights, vocab_size, **params)
	callback_list = get_callbacks(**params)

	X_train, Y_train, X_test, Y_test = load_XYs(**params)

	model = train_model(model, X_train, Y_train, callback_list, **params)
	
	evaluate_classification(X_train, Y_train,
							label = embedding_type,
							**params)
		
	evaluate_classification(X_test, Y_test,
							label = embedding_type,
							**params)