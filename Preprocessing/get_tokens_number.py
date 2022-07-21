

def get_tokens_number(X_tokenized, **params):
	print('calculating total tokens number...')
	
	number_of_tokens = 0

	for clause in X_tokenized:
	    for token in clause:
	        number_of_tokens += 1
 
	return number_of_tokens