import pickle

def load_cleaned_X(**params):

	print ("Trying to load_cleanedX....")

	with open('./Data/CleanedX.pkl', 'rb') as f:
		X = pickle.load(f)

	return X
