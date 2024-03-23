
from LPA import PCA, Corpus, sockpuppet_distance, prepare_for_visualization
import logging


# # Visualization and summary
# 
# hist_data = pd.DataFrame([sig.sum() for sig in signatures], index=corpus.document_cat.categories)
# hist_data.plot.hist()
#
# print(corpus.freq.head(5))

# logging.info("Calculating sockpuppet distance...")

# spd = sockpuppet_distance(corpus, corpus)

# logging.info(f"Sockpuppet distance calculated: {spd}")

# pca, evr = PCA(spd, n_components=2)

# logging.info("PCA completed.")
# logging.info(f"PCA result: {pca}")
# logging.info(f"Explained Variance Ratio: {evr}")

# logging.info("Program ended successfully.")


import altair as alt
import pandas as pd

from LPA import PCA, Corpus, sockpuppet_distance
from LPA import sockpuppet_distance, Corpus

def main():
    logging.basicConfig(filename='progress_log.txt', level=logging.INFO, 
                        format='%(asctime)s %(levelname)s:%(message)s')

    alt.data_transformers.disable_max_rows()
    
    logging.info("1.Reading frequency data...")
    freq = pd.read_csv('test_data/post_frequency.csv')
    logging.info("  Data loaded successfully.")
    
    logging.info("2. Creating DVR from the corpus...")
    corpus = Corpus(freq=freq, name='Corpus')
    dvr = corpus.create_dvr()
    logging.info("DVR created.")
    
    epsilon_frac = 2
    epsilon = 1 / (len(dvr) * epsilon_frac)
    logging.info(f"Epsilon calculated: {epsilon}")

    logging.info("Creating signatures...")
    signatures = corpus.create_signatures(epsilon=epsilon, sig_length=500, distance="KLDe")
    logging.info("Signatures created.")

    logging.info("Generating histogram...")
    pd.DataFrame(
        [sig.sum() for sig in signatures], index=corpus.document_cat.categories
    ).plot.hist()
    pd.DataFrame([sig.sum() for sig in signatures])

    logging.info("Calculating sockpuppet distance...")
    spd = sockpuppet_distance(corpus, corpus, res="matrix")
    logging.info(f"Sockpuppet distance calculated {spd}")

    # pca, evr = PCA(spd, n_components=2)
    # # print(plot_pca(pca,spd.inde

if __name__ == '__main__':
    main()