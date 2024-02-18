import altair as alt
import pandas as pd

from LPA import PCA, Corpus, sockpuppet_distance
from visualize import plot_pca, sockpuppet_matrix

alt.data_transformers.disable_max_rows()

# freq = pd.read_csv('./frequency.csv')
freq = pd.read_csv('test_data/post_frequency.csv')
# print(freq.head())
# print(freq.describe(include="all"))

corpus = Corpus(freq=freq)
dvr = corpus.create_dvr()
print(dvr)

# epsilon_frac = 2
# epsilon = 1 / (len(dvr) * epsilon_frac)
# # print(epsilon)

# signatures = corpus.create_signatures(epsilon=epsilon, sig_length=500, distance="KLDe")
# # print(signatures[0].head(10))

# pd.DataFrame(
#     [sig.sum() for sig in signatures], index=corpus.document_cat.categories
# ).plot.hist()
# # pd.DataFrame([sig.sum() for sig in signatures])

# spd = sockpuppet_distance(corpus, corpus)
# # print(spd.head())

# sockpuppet_matrix(spd)


# spd = sockpuppet_distance(corpus, corpus, res="matrix")
# pca, evr = PCA(spd, n_components=2)
# # print(plot_pca(pca,spd.inde