

def get_unique_words(X_tokenized, **params):
	print('get total unique words...')
	unique_words = []

	for clause in X_tokenized:

	  for token in clause:
	    if not token in unique_words:
	      unique_words.append(token)

	return unique_words