from src.utils.load_data import *
from src.utils.processing import *
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import pandas as pd

from timeit import default_timer as timer

async def main(N: int, N_UNIGRAMS: int, TRAIN_SIZE: int, TEST_SIZE: int, show_timer: bool = False):
	s_load = timer()
	data = await load_reviews(TRAIN_SIZE, TEST_SIZE)
	e_load = timer()

	s_process = timer()
	split_data = split_unigrams(data)
	set_data = get_unigram_sets(split_data)
	top_u = get_top_unigrams(N_UNIGRAMS, split_data)
	mtx = get_unigram_presence_matrices(set_data, top_u)

	x_train = np.concatenate([mtx['positive_train'], mtx['negative_train']])
	y_train = np.concatenate([np.ones(TRAIN_SIZE), np.zeros(TRAIN_SIZE)])
	x_test = np.concatenate([mtx['positive_test'], mtx['negative_test']])
	y_test = np.concatenate([np.ones(TEST_SIZE), np.zeros(TEST_SIZE)])
	e_process = timer()

	s_nb_train = timer()
	nb = GaussianNB()
	nb.fit(x_train, y_train)
	e_nb_train = timer()
	y_pred = nb.predict(x_test)
	n_correct = (y_test == y_pred).sum()
	print(f'NaiveBayes accuracy: {n_correct*100/x_test.shape[0]}% {n_correct}/{x_test.shape[0]}')

	s_svm_train = timer()
	clf = svm.SVC()
	clf.fit(x_train, y_train)
	e_svm_train = timer()
	y_pred = clf.predict(x_test)
	n_correct = (y_test == y_pred).sum()
	print(f"SVM accuracy: {n_correct*100/x_test.shape[0]}% {n_correct}/{x_test.shape[0]}")

	if show_timer:
		print(f"loading: {e_load - s_load}")
		print(f"processing: {e_process - s_process}s")
		print(f"NB training: {e_nb_train - s_nb_train}s")
		print(f"SVM training: {e_svm_train - s_svm_train}s")

if __name__ == "__main__":
	N = 1200
	N_UNIGRAMS = 3000

	TRAIN_SIZE = N*3//4
	TEST_SIZE = N//4
	asyncio.run(main(N, N_UNIGRAMS, TRAIN_SIZE, TEST_SIZE, True))