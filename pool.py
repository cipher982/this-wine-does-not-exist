from multiprocessing import Pool
from difflib import SequenceMatcher
import pandas as pd

import funcs

real_names_path = 'data/scraped/names_prices_descriptions.pickle'
fake_names_path = 'data/fake/fake_names_12949.pickle'

if __name__ == '__main__':
	with Pool(8) as p:
		scores = p.map(funcs.similarity, [(0, 1618),
										 (1618, 3237),
										 (3237, 4855),
										 (4855, 6474),
										 (6474, 8093),
										 (8093, 9711),
										 (9711, 11330),
										 (11330, 12949)])

	pd.to_pickle(scores, 'new_scores.pickle')
