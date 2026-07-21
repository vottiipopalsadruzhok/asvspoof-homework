import numpy as np

from src.metrics.base_metric import BaseMetric


class EERMetric(BaseMetric):
    """
    Equal Error Rate over a whole partition, in percent.

    The DET curve is traced exactly as in the official ASVspoof2019 scoring
    script, so the value matches the one used for grading.
    """

    def __init__(self, name="EER", *args, **kwargs):
        """
        Args:
            name (str): metric name to use in logger and writer.
        """
        super().__init__(*args, name=name, **kwargs)

    def __call__(self, scores, labels, **batch):
        """
        Args:
            scores (array-like): per-utterance bonafide scores.
            labels (array-like): per-utterance labels, 1 for bonafide.
        Returns:
            eer (float): equal error rate in percent.
        """
        scores = np.asarray(scores)
        labels = np.asarray(labels)

        frr, far = self._det_curve(scores[labels == 1], scores[labels == 0])
        crossing = np.argmin(np.abs(frr - far))

        return 100 * np.mean((frr[crossing], far[crossing]))

    @staticmethod
    def _det_curve(bonafide_scores, spoof_scores):
        """
        Sweep the decision threshold over the sorted scores and trace both
        error rates. Ties are resolved by a stable sort, which counts the
        bonafide trials of a tied group as rejected first.

        Args:
            bonafide_scores (ndarray): scores of the bonafide trials.
            spoof_scores (ndarray): scores of the spoof trials.
        Returns:
            frr (ndarray): false rejection rate at each threshold.
            far (ndarray): false acceptance rate at each threshold.
        """
        n_bonafide, n_spoof = bonafide_scores.size, spoof_scores.size

        scores = np.concatenate((bonafide_scores, spoof_scores))
        is_bonafide = np.concatenate((np.ones(n_bonafide), np.zeros(n_spoof)))
        is_bonafide = is_bonafide[np.argsort(scores, kind="mergesort")]

        rejected_bonafide = np.cumsum(is_bonafide)
        accepted_spoof = n_spoof - (
            np.arange(1, n_bonafide + n_spoof + 1) - rejected_bonafide
        )

        frr = np.concatenate((np.zeros(1), rejected_bonafide / n_bonafide))
        far = np.concatenate((np.ones(1), accepted_spoof / n_spoof))

        return frr, far
