import Levenshtein as lvs  # noqa: N813 alias naming not matching the module name consider Lvs instead.
import numpy as np


class SequenceSplitter:
    """
    Class for calculating distances between sequences based on the Levenshtein distance.
    """

    def __init__(self, train_seqs: list[str], test_seqs: list[str]):
        self.train_seqs = train_seqs
        self.test_seqs = test_seqs
        self.all_seqs = train_seqs + test_seqs
        self.data_len = len(self.all_seqs)
        self.distance_matrix = self.compute_distance_matrix(self.all_seqs)
        self.sim_matrix = 1.0 - self.distance_matrix

    def compute_distance_matrix(self, seqs: list[str]) -> np.array:
        """
        Computes the Levenshtein distance matrix for a list of sequences.
        """
        ndim = len(seqs)
        dist_matrix = np.ones((ndim, ndim))
        for i, s1 in enumerate(seqs):
            for j, s2 in enumerate(seqs):
                if j == i:
                    dist_matrix[i, j] = 0.0
                elif i > j:
                    dist_matrix[i, j] = dist_matrix[j, i]
                else:
                    dist_matrix[i, j] = 1.0 - lvs.ratio(s1, s2)
        return dist_matrix

    def plot_dist(self, similarity=False, figure_path="sequence_identity_plot_trial2.png"):
        """
        Calculates the intra-class and inter-class distances and plots histograms.
        """

        train_indices = range(len(self.train_seqs))
        mask_train = np.ix_(train_indices, train_indices)
        train_distances = (
            self.sim_matrix[mask_train] if similarity else self.distance_matrix[mask_train]
        )
        intra_train = np.triu(train_distances).flatten()
        intra_train = intra_train[np.nonzero(intra_train)]

        test_indices = range(len(self.train_seqs), self.data_len)
        mask_test = np.ix_(test_indices, test_indices)
        test_distances = (
            self.sim_matrix[mask_test] if similarity else self.distance_matrix[mask_test]
        )
        intra_test = np.triu(test_distances).flatten()
        intra_test = intra_test[np.nonzero(intra_test)]

        mask_inter = np.ix_(train_indices, test_indices)
        inter_distances = (
            self.sim_matrix[mask_inter] if similarity else self.distance_matrix[mask_inter]
        )
        inter_train_test = inter_distances.flatten()

        return intra_train, intra_test, inter_train_test
