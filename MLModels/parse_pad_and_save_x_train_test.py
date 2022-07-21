import pickle
import pandas as pd

from Preprocessing import load_tokenized_X
from Preprocessing import save_XYs

from ._parsing import parsing
from ._padding import padding
from ._train_test_split import train_test_split_
from ._oversample import oversample

from embedding import load_network_info


def parse_pad_and_save_x_train_test(Y, **params):

	print ("Trying to parse_pad_and_save_x_train_test...")

	index_dict, vocab_size, embedding_weights = \
			load_network_info(**params)

	X = load_tokenized_X(**params)
	X = parsing(X, index_dict, **params)
	X = padding(X, **params)

	X_train, X_test, Y_train, Y_test = train_test_split_(X, Y, **params)
	# X_train, Y_train = oversample(X_train, Y_train, **params)

	save_XYs(X_train, Y_train, X_test, Y_test, **params)