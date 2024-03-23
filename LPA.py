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


def calculate_block_distance(args):
    block, matrix2, start_row = args
    print(f"Processing block starting at row {start_row}")
    distances = cdist(block, matrix2, metric="cityblock")
    print(f"Finished processing block starting at row {start_row}")
    return start_row, distances

def sockpuppet_distance(corpus1, corpus2, res='table', heuristic=True):
    matrix1 = deepcopy(corpus1.signature_matrix.matrix)
    matrix2 = deepcopy(corpus2.signature_matrix.matrix)

    if heuristic:
        matrix1[matrix1 > 0] += 1
        matrix1[matrix1 < 0] -= 1
        matrix2[matrix2 > 0] += 1
        matrix2[matrix2 < 0] -= 1

    matrix1 = matrix1[:, ~np.all(matrix1 == 0, axis=0)]
    matrix2 = matrix2[:, ~np.all(matrix2 == 0, axis=0)]

    block_size = 1000  
    total_blocks = int(np.ceil(matrix1.shape[0] / block_size))
    print(f"Total blocks to process: {total_blocks}")

    cdist_ = np.zeros((matrix1.shape[0], matrix2.shape[0]))
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for start_row in range(0, matrix1.shape[0], block_size):
            block = matrix1[start_row:start_row + block_size]
            future = executor.submit(calculate_block_distance, (block, matrix2, start_row))
            futures.append(future)
            print(f"1 {future} " )
            print(f"Created process for block starting at row {start_row}")

        for future in futures:
            start_row, distances = future.result()
            cdist_[start_row:start_row + distances.shape[0], :] = distances
            print(f"Collected results for block starting at row {start_row}")

    c1n = getattr(corpus1, "name", "Corpus 1")
    c2n = getattr(corpus2, "name", "Corpus 2")
    df = pd.DataFrame(cdist_, index=corpus1.document_cat.categories, columns=corpus2.document_cat.categories)

    if res == "matrix":
        df /= df.values.max()
        df = df + df.T
        df = df.pivot(index=c1n, columns=c2n, values="value").fillna(0)
    else:
        df = df.melt(ignore_index=False, var_name=c2n).dropna().reset_index()
        df["value"] /= df["value"].max()

    return df

# def calculate_distances(block, matrix2, start_index, threshold=0.4):
#     distances = cdist(block, matrix2, metric='cityblock')
#     filtered_indices = np.where(distances < threshold)
#     filtered_distances = distances[filtered_indices]
#     # Adjust the indices to match the original matrix
#     filtered_row_indices = filtered_indices[0] + start_index
#     filtered_col_indices = filtered_indices[1]
    
#     # Return both the indices and the corresponding distances
#     return (filtered_row_indices, filtered_col_indices), filtered_distances


# def sockpuppet_distance(corpus1, corpus2, res='table', heuristic=True, threshold=0.4):
#     matrix1 = corpus1.signature_matrix.matrix
#     matrix2 = corpus2.signature_matrix.matrix

#     if heuristic:
#         matrix1 = np.where(matrix1 > 0, matrix1 + 1, matrix1)
#         matrix2 = np.where(matrix2 > 0, matrix2 + 1, matrix2)

#     matrix1 = matrix1[:, ~np.all(matrix1 == 0, axis=0)]
#     matrix2 = matrix2[:, ~np.all(matrix2 == 0, axis=0)]

#     block_size = 2000  # Adjust based on your system's capabilities
#     total_blocks = int(np.ceil(matrix1.shape[0] / block_size))

#     print(f"Total blocks to process: {total_blocks}")

#     spd_results = []
#     with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust the number of workers based on your system
#         futures = []
#         for start in tqdm(range(0, matrix1.shape[0], block_size), total=total_blocks, desc="Processing blocks"):
#             block = matrix1[start:start + block_size]
#             futures.append(executor.submit(calculate_distances, block, matrix2, start, threshold))

#         for future in tqdm(as_completed(futures), total=total_blocks, desc="Collecting results"):
#             indices, distances = future.result()
#             row_indices, col_indices = indices
#             for i in range(len(distances)):
#                 spd_results.append({
#                     corpus1.name: corpus1.document_cat.categories[row_indices[i]],
#                     corpus2.name: corpus2.document_cat.categories[col_indices[i]],
#                     'value': distances[i]
#                 })

#     spd_df = pd.DataFrame(spd_results)

#     return spd_df

# def calculate_distances(block, matrix2, start_index, threshold=0.4):
#     distances = cdist(block, matrix2, metric='cityblock')
#     # Filter the distances and get the indices for those less than the threshold
#     row_indices, col_indices = np.where(distances < threshold)
#     filtered_distances = distances[row_indices, col_indices]
#     # Adjust row indices to match their original position in the full matrix
#     row_indices += start_index
#     return row_indices, col_indices, filtered_distances

# def sockpuppet_distance(corpus1, corpus2, threshold=0.4):
#     matrix1 = corpus1.signature_matrix.matrix
#     matrix2 = corpus2.signature_matrix.matrix

#     # Ensure the matrices are filtered as per heuristic, if needed
#     matrix1 = np.where(matrix1 > 0, matrix1 + 1, matrix1)
#     matrix2 = np.where(matrix2 > 0, matrix2 + 1, matrix2)

#     matrix1 = matrix1[:, ~np.all(matrix1 == 0, axis=0)]
#     matrix2 = matrix2[:, ~np.all(matrix2 == 0, axis=0)]

#     block_size = 500
#     total_blocks = int(np.ceil(matrix1.shape[0] / block_size))

#     results = []
#     with ProcessPoolExecutor() as executor:
#         futures = []
#         for start in range(0, matrix1.shape[0], block_size):
#             block = matrix1[start:start + block_size]
#             futures.append(executor.submit(calculate_distances, block, matrix2, start, threshold))

#         for future in as_completed(futures):
#             row_indices, col_indices, distances = future.result()
#             results.extend(zip(row_indices, col_indices, distances))

#     # Create a DataFrame from the results
#     df = pd.DataFrame(results, columns=['row_index', 'col_index', 'distance'])

#     # Map indices to categories
#     df['document'] = df['row_index'].apply(lambda x: corpus1.document_cat.categories[x])
#     df['element'] = df['col_index'].apply(lambda x: corpus2.document_cat.categories[x])

#     # Drop the numerical indices as they are no longer needed
#     df = df.drop(['row_index', 'col_index'], axis=1)

#     return df.pivot(index='document', columns='element', values='distance')


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
