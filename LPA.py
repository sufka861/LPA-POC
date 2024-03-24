from __future__ import annotations
from importlib import import_module
from pathlib import Path
from typing import List, Tuple
from scipy.spatial.distance import cdist
from sklearn import decomposition as skd
from sklearn.preprocessing import StandardScaler
from algo import symmetrized_KLD
from helpers import write
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from copy import deepcopy

class Matrix:
    def __init__(self, matrix: np.array):
        self.matrix = matrix
        self.normalized = False

    def __bool__(self):
        return True if hasattr(self, "matrix") else False

    def epsilon_modification(
        self,
        epsilon: float | None = None,
        lambda_: float | int = 1,
        threshold: float = 0,
    ):
        if not epsilon:
            if epsilon == 0:
                return
            raise ValueError("Epsilon must be provided")
            # epsilon = self._get_epsilon(lambda_)
            # if epsilon == 0:
            #     return
        beta = 1 - epsilon * np.count_nonzero(self.matrix <= threshold, axis=1)
        self.matrix = self.matrix * beta[:, None]
        self.matrix[self.matrix <= threshold] = epsilon

    def apply(
        self, metric: str, save: bool = False, path: None | Path = None
    ) -> pd.DataFrame:
        res = []
        func = getattr(import_module("algo"), metric)
        # TODO: apply_along_axis or something
        for i in range(len(self.matrix) - 1):
            res.append(func(self.matrix[i : i + 2]))
        res_df = (
            pd.DataFrame({metric: res}).reset_index().rename(columns={"index": "date"})
        )
        if save:
            write(path, (res_df, metric))
        return res_df

    def delete(self, ix, axis):
        self.matrix = np.delete(self.matrix, obj=ix, axis=axis)

    def normalize(self):
        self.normalized = True
        self.matrix = (self.matrix.T / self.matrix.sum(axis=1)).T

    def create_dvr(self):
        if self.normalized:
            raise ValueError("Cannot create the DVR from normalized frequency data")
        self.dvr = self.normalized_weight()

    def normalized_weight(self) -> np.ndarray:
        return self.matrix.sum(axis=0) / self.matrix.sum()

    def moving_average(self, window: int) -> np.array:
        max_ = bn.nanmax(self.matrix, axis=1)
        min_ = bn.nanmin(self.matrix, axis=1)
        ma = bn.move_mean(bn.nanmean(self.matrix, axis=1), window=window, min_count=1)
        return pd.DataFrame({"ma": ma, "max": max_, "min": min_}).reset_index()


class Corpus:
    def __init__(
        self,
        freq: pd.DataFrame | None = None,
        document_cat: pd.Series | pd.DatetimeIndex | None = None,
        element_cat: pd.Series | None = None,
        name: str | None = None,
    ):
        if (
            isinstance(freq, type(None))
            and isinstance(document_cat, type(None))
            and isinstance(element_cat, type(None))
        ):
            raise ValueError(
                "Either use a frequency dataframe or two series, one of document ids and one of elements"
            )
        elif isinstance(freq, pd.DataFrame):
            self.freq = freq
            document_cat = freq["document"]
            element_cat = freq["element"]
        self.document_cat = pd.Categorical(document_cat, ordered=True).dtype
        self.element_cat = pd.Categorical(element_cat, ordered=True).dtype
        if name:
            self.name = name

    def __len__(self):
        """Number of documents"""
        return len(self.matrix.matrix)

    def current(self, m=True):
        if hasattr(self, "signature_matrix"):
            curr = self.signature_matrix
        elif hasattr(self, "distance_matrix"):
            curr = self.distance_matrix
        return curr.matrix if m else curr

    def update_documents(self, document):
        self.document_cat = pd.CategoricalDtype(
            self.document_cat.categories[
                ~self.document_cat.categories.isin([document])
            ],
            ordered=True,
        )

    def code_to_cat(self, code: str, what="document") -> int:
        return getattr(self, f"{what}_cat").categories[code]

    def pivot(self, freq: pd.DataFrame | None = None) -> Matrix:
        if hasattr(self, "freq"):
            freq = self.freq
        d = freq["document"].astype(self.document_cat)
        e = freq["element"].astype(self.element_cat)
        idx = np.array([d.cat.codes, e.cat.codes]).T
        matrix = np.zeros(
            (len(d.cat.categories), len(e.cat.categories)), dtype="float64"
        )
        matrix[idx[:, 0], idx[:, 1]] = freq["frequency_in_document"]
        return Matrix(matrix[min(d.cat.codes) : max(d.cat.codes) + 1])

    def create_dvr(self, matrix: None | Matrix = None) -> pd.DataFrame:
        if not matrix:
            self.matrix = self.pivot(self.freq)
            matrix = self.matrix
        matrix.create_dvr()
        dvr = (
            pd.DataFrame(
                {
                    "element": self.element_cat.categories,
                    "global_weight": matrix.dvr,
                }
            )
            .reset_index()
            .rename(columns={"index": "element_code"})
            .sort_values("global_weight", ascending=False)
            .reset_index(drop=True)
        )
        return dvr[["element", "global_weight"]]

    def _signature_matrix(self, sig_length, distances_df):
        # annuls all values that shouldn't appear in the signatures
        self.signature_matrix = Matrix(self.current().copy())  # copy?
        if sig_length:
            argsort = np.argsort(np.abs(self.signature_matrix.matrix), axis=1)
            indices = argsort[:, -sig_length:]
            p = np.zeros_like(self.signature_matrix.matrix)
            for i in range(p.shape[0]):
                p[i, indices[i]] = self.signature_matrix.matrix[i, indices[i]]
            self.signature_matrix.matrix = p
        signatures = [
            sig[1][self.signature_matrix.matrix[i] != 0].sort_values(
                key=lambda x: abs(x), ascending=False
            )
            for i, sig in enumerate(distances_df.iterrows())
        ]
        return signatures

    def create_signatures(
        self,
        epsilon: float | None = None,
        sig_length: int | None = 500,
        distance: str = "KLDe",
    ) -> List[pd.DataFrame] | Tuple[List[pd.DataFrame]]:
        """
        most_significant: checks which elements had the largest distance altogether and returns a dataframe consisting only of those distances, sorted
        """
        if sig_length == 0:
            sig_length = None
        if not hasattr(self, "matrix"):
            raise AttributeError("Please create dvr before creating signatures.")
        if not self.matrix.normalized:
            self.matrix.normalize()
        if distance == "KLDe":
            self.matrix.epsilon_modification(epsilon)
        dm = symmetrized_KLD(self.matrix.matrix, self.matrix.dvr)
        self.distance_matrix = Matrix(dm)
        distances_df = pd.DataFrame(
            self.current(),
            index=self.document_cat.categories,
            columns=self.element_cat.categories,
        )
        res = self._signature_matrix(sig_length, distances_df)
        return res

# def calculate_block_distance(args):
#     block, matrix2, start_row, threshold = args
#     distances = cdist(block, matrix2, metric="cityblock")
#     # Keep only distances less than the threshold
#     filtered_indices = np.where(distances < threshold)
#     distances[filtered_indices] = np.nan  # Mark filtered distances as NaN
#     return start_row, distances

# def sockpuppet_distance(corpus1, corpus2, threshold=0.4, res='table', heuristic=True):
#     matrix1 = deepcopy(corpus1.signature_matrix.matrix)
#     matrix2 = deepcopy(corpus2.signature_matrix.matrix)

#     if heuristic:
#         matrix1[matrix1 > 0] += 1
#         matrix1[matrix1 < 0] -= 1
#         matrix2[matrix2 > 0] += 1
#         matrix2[matrix2 < 0] -= 1

#     matrix1 = matrix1[:, ~np.all(matrix1 == 0, axis=0)]
#     matrix2 = matrix2[:, ~np.all(matrix2 == 0, axis=0)]

#     block_size = 1000  
#     total_blocks = int(np.ceil(matrix1.shape[0] / block_size))
#     print(f"Total blocks to process: {total_blocks}")

#     cdist_ = np.zeros((matrix1.shape[0], matrix2.shape[0]))

#     with ProcessPoolExecutor(max_workers=4) as executor:
#         futures = []
#         for start_row in range(0, matrix1.shape[0], block_size):
#             block = matrix1[start_row:start_row + block_size]
#             future = executor.submit(calculate_block_distance, (block, matrix2, start_row, threshold))
#             futures.append(future)

#         for future in futures:
#             start_row, distances = future.result()
#             cdist_[start_row:start_row + distances.shape[0], :] = distances

#     if res == "matrix":
#         df = pd.DataFrame(cdist_, index=corpus1.document_cat.categories, columns=corpus2.document_cat.categories)
#         if heuristic:  # Optional symmetry step
#             df = df + df.T
#         df /= df.max().max()
#     else:
#         df = pd.DataFrame(cdist_).stack().reset_index()
#         df.columns = ['row', 'col', 'value']
#         df = df[df['value'] < threshold]

#     return df




# def calculate_block_distance(block, matrix2, metric="cityblock"):
#     return cdist(block, matrix2, metric=metric)

# def sockpuppet_distance(
#     corpus1,
#     corpus2,
#     res: Literal["table", "matrix"] = "table",
#     heuristic: bool = True,
#     threshold = 0.02
# ):
#     # Preprocess matrices
#     matrices = []
#     for corpus in [corpus1, corpus2]:
#         matrix = deepcopy(corpus.signature_matrix.matrix)
#         matrix = matrix[:, ~np.all(matrix == 0, axis=0)]
#         if heuristic:
#             matrix[matrix > 0] += 1
#             matrix[matrix < 0] -= 1
#         matrices.append(matrix)

#     matrix1, matrix2 = matrices
#     block_size = 700
#     total_rows = matrix1.shape[0]

#     # Calculate max value for filtering threshold before normalization
#     sample_distance = cdist(matrix1[:1], matrix2, metric="cityblock")
#     max_value = sample_distance.max()
#     raw_threshold = threshold * max_value

#     cdist_ = np.zeros((total_rows, matrix2.shape[0]))
    
#     total_blocks = int(np.ceil(total_rows / block_size))
#     print(f"Total blocks to process: {total_blocks}")

#     with ProcessPoolExecutor(max_workers=4) as executor:
#         futures = []
#         for start_row in range(0, total_rows, block_size):
#             end_row = min(start_row + block_size, total_rows)
#             block = matrix1[start_row:end_row]
#             future = executor.submit(calculate_block_distance, block, matrix2)
#             futures.append((future, start_row, end_row))

#         for future, start_row, end_row in futures:
#             block_distances = future.result()
#             if corpus1 is corpus2:
#                 # Set diagonal to 0 if comparing the same corpus
#                 np.fill_diagonal(block_distances[max(0, start_row - end_row):], 0)
#             block_distances[block_distances > raw_threshold] = np.nan
#             cdist_[start_row:end_row, :] = block_distances

#     # Normalization
#     cdist_ /= max_value

#     # Construct DataFrame
#     c1n = getattr(corpus2, "name", "Corpus 1")
#     c2n = getattr(corpus1, "name", "Corpus 2")
#     df = pd.DataFrame(cdist_, index=corpus1.document_cat.categories, columns=corpus2.document_cat.categories)

#     if res == "table":
#         if c1n == c2n:
#             c2n = c1n + " "
#         df = df.rename_axis(index=c1n).melt(ignore_index=False, var_name=c2n).dropna().reset_index()
#     elif res == "matrix":
#         df = df.fillna(0) + df.T

#     return df


def sockpuppet_distance(
    corpus1,
    corpus2,
    res: Literal["table", "matrix"] = "table",
    heuristic: bool = True
):
    # Preprocess matrices
    matrices = []
    for corpus in [corpus1, corpus2]:
        matrix = deepcopy(corpus.signature_matrix.matrix)
        matrix = matrix[:, ~np.all(matrix == 0, axis=0)]
        if heuristic:
            matrix[matrix > 0] += 1
            matrix[matrix < 0] -= 1
        matrices.append(matrix)

    matrix1, matrix2 = matrices
    block_size = 700

    # Initialize distance matrix
    total_rows = matrix1.shape[0]
    cdist_ = np.zeros((total_rows, matrix2.shape[0]))

    # Calculate distances in blocks for upper triangle
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for start_row in range(0, total_rows, block_size):
            end_row = min(start_row + block_size, total_rows)
            block1 = matrix1[start_row:end_row]
            for start_col in range(start_row, matrix2.shape[0], block_size):
                end_col = min(start_col + block_size, matrix2.shape[0])
                block2 = matrix2[start_col:end_col]
                future = executor.submit(calculate_block_distance, block1, block2)
                futures.append((future, start_row, end_row, start_col, end_col))

        for future, start_row, end_row, start_col, end_col in futures:
            block_distances = future.result()
            cdist_[start_row:end_row, start_col:end_col] = block_distances

    # Make the matrix symmetric
    cdist_ = np.triu(cdist_) + np.triu(cdist_, 1).T

    # Normalize the distance matrix
    max_value = cdist_.max() if cdist_.max() > 0 else 1  # Prevent division by zero
    cdist_ /= max_value

    # Construct DataFrame
    c1n = getattr(corpus1, "name", "Corpus 1")
    c2n = getattr(corpus2, "name", "Corpus 2")
    df = pd.DataFrame(cdist_, index=corpus1.document_cat.categories, columns=corpus2.document_cat.categories)

    if res == "table":
        if c1n == c2n:
            c2n = c1n + " "
        df = df.rename_axis(index=c1n).melt(ignore_index=False, var_name=c2n).dropna().reset_index()
    elif res == "matrix":
        df = df.fillna(0)

    return df

def calculate_block_distance(block1, block2):
    return cdist(block1, block2, metric="cityblock")

def PCA(sockpuppet_matrix, n_components: int = 2):
    """
    Creates a PCA object and returns it, as well as the explained variance ratio.
    """
    scaler = StandardScaler()
    sockpuppet_matrix = scaler.fit_transform(sockpuppet_matrix)
    scaled_matrix = scaler.fit_transform(sockpuppet_matrix)
    pca = skd.PCA(n_components=n_components)
    pca.fit(scaled_matrix)
    res = pca.transform(scaled_matrix)
    return res, pca.explained_variance_ratio_

def prepare_for_visualization(spd_matrix):
    # Reset index to ensure the document identifiers are part of the DataFrame's data
    spd_matrix = spd_matrix.reset_index()
    # Melt the DataFrame to long format
    spd_long = spd_matrix.melt(id_vars=spd_matrix.columns[0], var_name='element', value_name='value')
    spd_long.rename(columns={spd_matrix.columns[0]: 'document'}, inplace=True)
    return spd_long
