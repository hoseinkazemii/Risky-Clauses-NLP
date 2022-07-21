import pickle

def load_tokenized_X(**params):

	print ("Trying to load_tokenized_X....")

	with open('./Data/TokenizedX.pkl', 'rb') as f:
		X = pickle.load(f)

	return X