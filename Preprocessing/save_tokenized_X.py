import pickle


def save_tokenized_X(X, **params):

	print ("Trying to pickle the TokenizedX...")

	with open('./Data/TokenizedX.pkl', 'wb') as f:
		pickle.dump(X, f)