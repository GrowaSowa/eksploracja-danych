import pandas as pd
import numpy as np
import sys
import os
import random
import asyncio

async def load_reviews(n_train: int, n_test)->dict[str, list[str]]:
	n = n_train + n_test
	negative, positive = await asyncio.gather(load_negative(n), load_positive(n))
	return {
		"negative_train": negative[:n_train],
		"negative_test": negative[n_train : n],
		"positive_train": positive[:n_train],
		"positive_test": positive[n_train : n]
	}

async def load_positive(n: int)->list[str]:
	train_pos = ['./data/aclImdb/train/pos/' + fname for fname in os.listdir('./data/aclImdb/train/pos')]
	test_pos = ['./data/aclImdb/test/pos/'+fname for fname in os.listdir('./data/aclImdb/test/pos')]

	positive = train_pos
	positive.extend(test_pos)
	random.shuffle(positive)

	content = [get_content(fname) for fname in positive[:n]]
	return content

async def load_negative(n: int)->list[str]:
	train_neg = ['./data/aclImdb/train/neg/' + fname for fname in os.listdir('./data/aclImdb/train/neg')]
	test_neg = ['./data/aclImdb/test/neg/' + fname for fname in os.listdir('./data/aclImdb/test/neg')]

	negative = train_neg
	negative.extend(test_neg)
	random.shuffle(negative)

	content = [get_content(fname) for fname in negative[:n]]
	return content

def get_content(fname: str)->str:
	with open(fname, 'r', encoding='utf-8') as f:
		return ' '.join(f.readlines())