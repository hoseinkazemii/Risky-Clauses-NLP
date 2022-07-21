from keras.preprocessing.text import text_to_word_sequence


def tokenize(X, **params):
	print('tokenizing X...')
	X_tokenized = []

	for clause in X:
		X_tokenized.append(text_to_word_sequence(clause))

	return X_tokenized