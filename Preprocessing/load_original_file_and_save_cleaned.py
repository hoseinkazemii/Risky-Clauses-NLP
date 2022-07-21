import pickle

from .load_data import load_data
from .split_data import split_data
from .remove_stop_words import remove_stop_words


def load_original_file_and_save_cleaned(**params):

	df = load_data(**params)
	X, _ = split_data(df, **params)
	# X = remove_stop_words(X, **params)

	print ("Trying to pickle the CleanedX...")
	with open('./Data/CleanedX.pkl', 'wb') as f:
		pickle.dump(X, f)

	print ("Cleaned file is saved...")