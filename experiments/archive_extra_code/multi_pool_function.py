from difflib import SequenceMatcher
import pandas as pd


def similarity(indices):
    real_names = pd.read_pickle("real_name_list.pickle")
    fake_names_path = "data/fake/fake_names_12949.pickle"
    fake_names = pd.read_pickle(fake_names_path)
    fake_names = fake_names[indices[0] : indices[1]]

    scores = {}
    for i, fake_name in enumerate(fake_names):
        max_score = 0.0
        for real_name in real_names:
            similarity_score = SequenceMatcher(None, fake_name, real_name).ratio()
            if similarity_score > max_score:
                max_score = similarity_score
        scores[fake_name] = max_score
        print("i:", i, "\tcomplete:", float(i) / float(len(fake_names)))
    pd.to_pickle(scores, "scores_part_{}.pickle".format(str(indices)))
    return scores

