from src.utils.load_data import *
from src.utils.processing import *
from sklearn.naive_bayes import GaussianNB
import pandas as pd

from timeit import default_timer as timer

N = 1000
N_UNIGRAMS = 3000

async def main():
	s_load = timer()
	data = await load_reviews(N)
	e_load = timer()

	s_process = timer()
	split_data = split_unigrams(data)

	set_data = get_unigram_sets(split_data)

	top_u = get_top_unigrams(N_UNIGRAMS, split_data)

	mtx = get_unigram_presence_matrices(set_data, top_u)
	e_process = timer()

	TRAIN_SIZE = N*3//4
	TEST_SIZE = N//4

	s_split = timer()
	x_train = mtx['positive'][:TRAIN_SIZE]
	x_train.extend(mtx['negative'][:TRAIN_SIZE])
	y_train = [1 if i < TRAIN_SIZE else 0 for i in range(TRAIN_SIZE*2)]

	x_test = mtx['positive'][TRAIN_SIZE : TRAIN_SIZE+TEST_SIZE]
	x_test.extend(mtx['negative'][TRAIN_SIZE : TRAIN_SIZE+TEST_SIZE])
	y_test = [1 if i < TEST_SIZE else 0 for i in range(TEST_SIZE*2)]

	x_train = pd.DataFrame(x_train)
	y_train = np.array(y_train)

	x_test = pd.DataFrame(x_test)
	y_test = np.array(y_test)
	e_split = timer()

	s_train = timer()
	nb = GaussianNB()
	nb.fit(x_train, y_train)
	e_train = timer()
	y_pred = nb.predict(x_test)
	n_correct = (y_test == y_pred).sum()
	print(f'Accuracy: {n_correct*100/x_test.shape[0]}% {n_correct}/{x_test.shape[0]}')

	print(f"loading: {e_load - s_load}")
	print(f"processing: {e_process - s_process}")
	print(f"splitting: {e_split - s_split}")
	print(f"training: {e_train - s_train}")

if __name__ == "__main__":
	asyncio.run(main())