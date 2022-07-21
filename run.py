from Preprocessing import *
from embedding import *
from MLModels import *
from BERT import train_bert_and_report


def run():
	general_settings = {
	'embedding_type':'fasttext',
	'emb_dimension':300,
	'window_size':5,
	'min_word_count_wv':3,
	'skipgram':1,
	'wv_epochs':5,
	'maxlen':50,
	'n_cores':6,
	'split_size':0.2,
	'random_state':42,
	'k_nbrs_overs':5,
	'train_epochs':10,
	'batch_size':32,
	'val_split':0.2,
	'model_verbose':2,
	'lr':0.001,
	'pretrained_emb':True,
	'dropout':0.2,
	'rec_dropout':0.25,
	'should_warm_up':False,
	'network_type':"LSTM",
	'layers':[64,64],
	'dense_layers':[32,16],
	'checkpoint':"HooshvareLab/bert-fa-base-uncased",
	"bert_padding" : 64,
	"lr_bert": 0.00002,
	"bert_epochs" : 3,




	}

	# One time Run (Preprocessing):

	df = load_data(**general_settings)
	X, Y = split_data(df, **general_settings)
	X = tokenize(X, **general_settings)
	save_tokenized_X(X, **general_settings)
	number_of_tokens = get_tokens_number(X, **general_settings)
	unique_words = get_unique_words(X, **general_settings)
	count = count_words(X, **general_settings)
	create_word2vec_embedding(**general_settings)
	# create_fasttext_embedding(**general_settings)
	X = remove_stop_words(X, **general_settings)
	parse_pad_and_save_x_train_test(Y, **general_settings)


	# Training DNNs:
	train_model_and_report(**general_settings)


	# Bert
	load_original_file_and_save_cleaned(**general_settings)

	train_bert_and_report(**general_settings)









if __name__ == '__main__':
	run()




