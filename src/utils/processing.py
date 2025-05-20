import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

REMOVE_RE = re.compile(' *<br \/> *')
SEPARATE_RE = re.compile(' *(\.|!\?|!|\?|,) *')
SPACES_RE = re.compile(' {2,}')
ADDITIONAL_STOPWORDS = ['movie', 'film', ',', '-']

def split_unigrams(data: dict[str, list[str]])->dict[str, list[list[str]]]:
	stopwords = get_stopwords()
	lemmatizer = WordNetLemmatizer()
	
	def process_tokens(line: str):
		r = re.sub(REMOVE_RE, ' ', line)
		r = re.sub(SEPARATE_RE, lambda m: f' {m[0].strip()} ', r).strip()
		r = re.sub(SPACES_RE, ' ', r).split(' ')
		return [lemmatizer.lemmatize(unigram.lower()) for unigram in r if unigram.lower() not in stopwords]

	return {
		"positive": [process_tokens(line) for line in data['positive']],
		"negative": [process_tokens(line) for line in data['negative']]
	}


def get_top_unigrams(n: int, token_list_dict: dict[str, list[list[str]]])->list[str]:
	unigrams = {}

	for key in token_list_dict:
		for token_list in token_list_dict[key]:
			for token in token_list:
				if token in unigrams:
					unigrams[token] += 1
				else:
					unigrams[token] = 1
	
	ulist = list(unigrams.items())
	ulist.sort(key=lambda t: t[1], reverse=True)
	return [item[0] for item in ulist[:n]] if n>0 else [item[0] for item in ulist]

def get_unigram_sets(token_list_dict: dict[str, list[list[str]]])->dict[str, list[set[str]]]:
	return {
		"positive": [set(token_list) for token_list in token_list_dict['positive']],
		"negative": [set(token_list) for token_list in token_list_dict['negative']]
	}

def get_unigram_presence_matrix(token_set: set[str], top_unigrams: list[str])->np.ndarray:
	return np.array([int(unigram in token_set) for unigram in top_unigrams])


def get_unigram_presence_matrices(token_set_dict: dict[str, list[set[str]]], top_unigrams: list[str])->dict[str, list[np.ndarray]]:
	return {
		"positive": [get_unigram_presence_matrix(tset, top_unigrams) for tset in token_set_dict['positive']],
		"negative": [get_unigram_presence_matrix(tset, top_unigrams) for tset in token_set_dict['negative']]
	}

def get_stopwords(download: bool = False):
	if download:
		nltk.download('stopwords')
	stopwords = list(nltk.corpus.stopwords.words('english'))
	stopwords.extend(ADDITIONAL_STOPWORDS)
	return stopwords

def download_lemmatizer():
	nltk.download('wordnet')
	nltk.download('averaged_perceptron_tagger')