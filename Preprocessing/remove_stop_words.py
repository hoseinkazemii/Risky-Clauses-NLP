import json

from utils import ParallelProcess

from ._remove_stop_words_for_parallel import remove_stop_words_for_parallel

def remove_stop_words(X, **params):
	"""
	X: list of list
	"""
	print("removing stop words...")

	with open("./Preprocessing/_text_cleaners/persian_stopwords.json",
				'r', encoding = 'utf-8-sig') as f:
		stop_words = json.load(f)	

	n_cores = params.get("n_cores")

	results = ParallelProcess(X,
								remove_stop_words_for_parallel,
								stop_words,
								n_cores = n_cores)

	return results