import re

REMOVE_RE = '<br \/>'
SEPARATE_RE = '\.|!\?|!|\?'

def split_unigrams(data: dict[str, list[str]])->dict[str, list[list[str]]]:
	def process_tokens(line: str):
		r = re.sub(REMOVE_RE, '', line)
		return re.sub(SEPARATE_RE, lambda m: f' {m[0]}', r).split(' ')

	return {
		"positive": [process_tokens(line) for line in data['positive']],
		"negative": [process_tokens(line) for line in data['negative']]
	}


def get_top_unigrams(n: int, data: dict[str, list[list[str]]])->dict[str, int]:
	unigrams = {}

	for key in data:
		for token_list in data[key]:
			for token in token_list:
				if token in unigrams:
					unigrams[token] += 1
				else:
					unigrams[token] = 1
	
	ulist = list(unigrams.items())
	ulist.sort(key=lambda t: t[1], reverse=True)
	return ulist[:n] if n>0 else ulist
