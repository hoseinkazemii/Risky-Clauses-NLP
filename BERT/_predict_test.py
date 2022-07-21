import numpy as np
from sklearn.metrics import classification_report
import torch

def predict_test(model, X_test_seq, X_test_mask, Y_test, **params):
	device = torch.device("cpu")

	path = 'saved_weights.pt'
	model.load_state_dict(torch.load(path))

	with torch.no_grad():
		preds = model(X_test_seq.to(device), X_test_mask.to(device))
		preds = preds.detach().cpu().numpy()

	preds = np.argmax(preds, axis = 1)
	print(classification_report(Y_test, preds))
