def remove_stop_words_for_parallel(X,
								stop_words):

	for sw in stop_words:
		for sentence in X:

			if isinstance(sentence, str):
				sentence.replace(sw, "")

			else:
				try:
					while True:
						sentence.remove(sw)
				except ValueError:
					pass

	return X