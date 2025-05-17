import pandas as pd
import numpy as np
import sys
import os
import random

def load_ratings()->pd.DataFrame:
	ratings = pd.read_csv('./data/rating.csv',
		dtype={
			'userId': np.int64,
			'movieId': np.int64,
			'rating': np.float64,
			'timestamp': np.int64
		})
	ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
	return ratings

def load_movies()->pd.DataFrame:
	movies = pd.read_csv('./data/movie.csv',
		dtype={
			'movieId': np.int64,
			'title': str,
			'genres': str
		},
		index_col='movieId')
	movies['genres'] = movies['genres'].str.split('|')
	return movies

def load_tags()->pd.DataFrame:
	tags = pd.read_csv('./data/tag.csv',
		dtype={
			'userId': np.int64,
			'movieId': np.int64,
			'tag': str,
			'timestamp': np.int64
		})
	tags['timestamp'] = pd.to_datetime(tags['timestamp'], unit='s')
	return tags

def load_links()->pd.DataFrame:
	return pd.read_csv('./data/link.csv',
		dtype={
			'movieId': np.int64,
			'imdbId': np.int64,
			'tmbdId': np.int64
		},
		index_col='movieId')

def load_genome_tags()->pd.DataFrame:
	return pd.read_csv('./data/genome-tags.csv',
		dtype={
			'tagId': np.int64,
			'tag': str
		},
		index_col='tagId')

def load_genome_scores()->pd.DataFrame:
	return pd.read_csv('./data/genome-scores.csv',
		dtype={
			'movieId': np.int64,
			'tagId': np.int64,
			'relevance': np.float64
		})

def load_positive(n: int)->list[str]:
	train_pos = ['./data/aclImdb/train/pos/' + fname for fname in os.listdir('./data/aclImdb/train/pos')]
	test_pos = ['./data/aclImdb/test/pos/'+fname for fname in os.listdir('./data/aclImdb/test/pos')]

	positive = train_pos.copy()
	positive.extend(test_pos)
	random.shuffle(positive)
	
	content = [get_content(fname) for fname in positive[:n]]
	return content

def load_negative(n: int)->list[str]:
	train_neg = ['./data/aclImdb/train/neg/' + fname for fname in os.listdir('./data/aclImdb/train/neg')]
	test_neg = ['./data/aclImdb/test/neg/' + fname for fname in os.listdir('./data/aclImdb/test/neg')]

	negative = train_neg.copy()
	negative.extend(test_neg)
	random.shuffle(negative)

	content = [get_content(fname) for fname in negative[:n]]
	return content

def get_content(fname: str)->str:
	with open(fname, 'r') as f:
		return ' '.join(f.readlines())