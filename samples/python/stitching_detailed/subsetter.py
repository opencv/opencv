from itertools import chain
import cv2 as cv
import numpy as np

from stitching_detailed.feature_matcher import get_matches_matrix


class Subsetter:

    DEFAULT_CONFIDENCE_THRESHOLD = 1

    def __init__(self, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold

    def subset(self, features, pairwise_matches):
        indices = self.get_indices_to_keep(features, pairwise_matches)
        indices_to_delete = self.get_indices_to_delete(len(features),
                                                       indices)
        feature_subset = Subsetter.subset_list(features, indices)
        matches_matrix = get_matches_matrix(pairwise_matches)
        matches_matrix_subset = Subsetter.subset_matrix(matches_matrix,
                                                        indices_to_delete)
        matches_subset = Subsetter.__matrix_rows_to_list(matches_matrix_subset)

        return indices, feature_subset, matches_subset

    def get_indices_to_keep(self, features, pairwise_matches):
        indices = cv.detail.leaveBiggestComponent(features,
                                                  pairwise_matches,
                                                  self.confidence_threshold)
        return [int(idx) for idx in list(indices[:, 0])]

    def get_indices_to_delete(self, list_lenght, indices_to_keep):
        return list(set(range(list_lenght)) - set(indices_to_keep))

    @staticmethod
    def subset_list(list_to_subset, indices):
        return [list_to_subset[i] for i in indices]

    @staticmethod
    def subset_matrix(matrix_to_subset, indices_to_delete):
        for idx, idx_to_delete in enumerate(indices_to_delete):
            matrix_to_subset = Subsetter.__delete_index_from_matrix(
                matrix_to_subset,
                idx_to_delete-idx  # matrix shape reduced by one at each step
                )

        return matrix_to_subset

    def __delete_index_from_matrix(matrix, idx):
        mask = np.ones(matrix.shape[0], bool)
        mask[idx] = 0
        return matrix[mask, :][:, mask]

    def __matrix_rows_to_list(matrix):
        return list(chain.from_iterable(matrix.tolist()))
