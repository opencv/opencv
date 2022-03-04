from itertools import chain
import math
import cv2 as cv
import numpy as np

from .feature_matcher import FeatureMatcher
from .stitching_error import StitchingError


class Subsetter:

    DEFAULT_CONFIDENCE_THRESHOLD = 1
    DEFAULT_MATCHES_GRAPH_DOT_FILE = None

    def __init__(self,
                 confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                 matches_graph_dot_file=DEFAULT_MATCHES_GRAPH_DOT_FILE):
        self.confidence_threshold = confidence_threshold
        self.save_file = matches_graph_dot_file

    def subset(self, img_names, img_sizes, imgs, features, matches):
        self.save_matches_graph_dot_file(img_names, matches)
        indices = self.get_indices_to_keep(features, matches)

        img_names = Subsetter.subset_list(img_names, indices)
        img_sizes = Subsetter.subset_list(img_sizes, indices)
        imgs = Subsetter.subset_list(imgs, indices)
        features = Subsetter.subset_list(features, indices)
        matches = Subsetter.subset_matches(matches, indices)
        return img_names, img_sizes, imgs, features, matches

    def save_matches_graph_dot_file(self, img_names, pairwise_matches):
        if self.save_file:
            with open(self.save_file, 'w') as filehandler:
                filehandler.write(self.get_matches_graph(img_names,
                                                         pairwise_matches)
                                  )

    def get_matches_graph(self, img_names, pairwise_matches):
        return cv.detail.matchesGraphAsString(img_names, pairwise_matches,
                                              self.confidence_threshold)

    def get_indices_to_keep(self, features, pairwise_matches):
        indices = cv.detail.leaveBiggestComponent(features,
                                                  pairwise_matches,
                                                  self.confidence_threshold)

        if len(indices) < 2:
            raise StitchingError("No match exceeds the "
                                 "given confidence theshold.")

        return indices

    @staticmethod
    def subset_list(list_to_subset, indices):
        return [list_to_subset[i] for i in indices]

    @staticmethod
    def subset_matches(pairwise_matches, indices):
        indices_to_delete = Subsetter.get_indices_to_delete(
            math.sqrt(len(pairwise_matches)),
            indices
            )

        matches_matrix = FeatureMatcher.get_matches_matrix(pairwise_matches)
        matches_matrix_subset = Subsetter.subset_matrix(matches_matrix,
                                                        indices_to_delete)
        matches_subset = Subsetter.matrix_rows_to_list(matches_matrix_subset)

        return matches_subset

    @staticmethod
    def get_indices_to_delete(nr_elements, indices_to_keep):
        return list(set(range(int(nr_elements))) - set(indices_to_keep))

    @staticmethod
    def subset_matrix(matrix_to_subset, indices_to_delete):
        for idx, idx_to_delete in enumerate(indices_to_delete):
            matrix_to_subset = Subsetter.delete_index_from_matrix(
                matrix_to_subset,
                idx_to_delete-idx  # matrix shape reduced by one at each step
                )

        return matrix_to_subset

    @staticmethod
    def delete_index_from_matrix(matrix, idx):
        mask = np.ones(matrix.shape[0], bool)
        mask[idx] = 0
        return matrix[mask, :][:, mask]

    @staticmethod
    def matrix_rows_to_list(matrix):
        return list(chain.from_iterable(matrix.tolist()))
