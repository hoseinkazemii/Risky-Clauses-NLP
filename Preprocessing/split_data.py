

def split_data(df, **params):
	print('splitting data...')
	
	X = df['clause']
	y = df['risk']

	return X, y