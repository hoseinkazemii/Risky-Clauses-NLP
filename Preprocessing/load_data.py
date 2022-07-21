import pandas as pd

def load_data(**params):
	print('loading Dataset.csv...')
	df = pd.read_csv('./Data/Dataset.csv', encoding = 'utf-8')

	return df