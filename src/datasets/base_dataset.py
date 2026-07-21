import random

import torchaudio
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base class for audio datasets.

    A nested class only has to build an index (list[dict]) with a 'path' and a
    'label' per element; loading, the front-end and the rest are shared. Any
    other field of an index entry is passed through into the instance dict, so
    collate_fn and the submission writer can reach it by name.
    """

    def __init__(
        self,
        index,
        target_sr=16000,
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
        cache=False,
    ):
        """
        Args:
            index (list[dict]): metadata of each dataset element. The keys
                'path' and 'label' are required.
            target_sr (int): sample rate the audio is required to have.
            limit (int | None): if not None, keep only the first 'limit'
                elements (after the shuffle).
            shuffle_index (bool): if True, shuffle the index with a fixed
                seed (42).
            instance_transforms (dict[Callable] | None): transforms applied to
                a single element, keyed by tensor name. The special key
                "get_spectrogram" holds the waveform-to-spectrogram front-end.
            cache (bool): if True, keep processed elements in RAM. Random
                instance transforms are then frozen after the first epoch.
        """
        self._assert_index_is_valid(index)
        self._index: list[dict] = self._shuffle_and_limit_index(
            index, limit, shuffle_index
        )

        self.target_sr = target_sr
        self.instance_transforms = instance_transforms

        self._cache = {} if cache else None

    def __getitem__(self, ind):
        """
        Load the ind-th element and pack it into a dict.

        Key names must stay consistent across getitem, collate_fn, and the
        forward methods of the model and the loss.

        Args:
            ind (int): index in self._index.
        Returns:
            instance_data (dict): the metadata of the index entry plus the
                waveform, the spectrogram and the target label.
        """
        if self._cache is not None and ind in self._cache:
            return self._cache[ind]

        instance_data = dict(self._index[ind])
        instance_data["audio"] = self.load_audio(instance_data["path"])
        instance_data["labels"] = instance_data["label"]
        instance_data = self.preprocess_data(instance_data)

        if self._cache is not None:
            self._cache[ind] = instance_data

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def load_audio(self, path):
        """
        Load audio from disk and keep the first channel only.

        The sample rate is asserted instead of resampled: the front-end
        window and hop are defined in milliseconds at target_sr.

        Args:
            path (str): path to the audio file.
        Returns:
            audio (Tensor): waveform of shape (1, T), T varies between
                utterances.
        """
        audio, sr = torchaudio.load(path)
        assert sr == self.target_sr, f"Expected {self.target_sr} Hz, got {sr}: {path}"
        return audio[0:1, :]

    def get_spectrogram(self, audio):
        """
        Special instance transform with a special key to
        get spectrogram from audio.

        Args:
            audio (Tensor): original audio.
        Returns:
            spectrogram (Tensor): spectrogram for the audio.
        """
        return self.instance_transforms["get_spectrogram"](audio)

    def preprocess_data(self, instance_data):
        """
        Compute the spectrogram and apply the spectrogram-level transforms.

        Args:
            instance_data (dict): a single dataset element.
        Returns:
            instance_data (dict): the same element with the spectrogram added.
        """
        if self.instance_transforms is None:
            return instance_data

        instance_data["spectrogram"] = self.get_spectrogram(instance_data["audio"])

        if "spectrogram" in self.instance_transforms:
            instance_data["spectrogram"] = self.instance_transforms["spectrogram"](
                instance_data["spectrogram"]
            )

        return instance_data

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check that the index provides everything the base class relies on.

        Args:
            index (list[dict]): index to check.
        """
        for entry in index:
            assert (
                "path" in entry
            ), "Each dataset item should include field 'path' - path to audio file."
            assert (
                "label" in entry
            ), "Each dataset item should include field 'label' - object ground-truth label."

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Shuffle a copy of the index and cut it down to 'limit' elements.

        The shuffle uses its own Random(42), so the permutation neither
        depends on nor disturbs the global random state.

        Args:
            index (list[dict]): index to process.
            limit (int | None): if not None, number of elements to keep.
            shuffle_index (bool): if True, shuffle before limiting.
        Returns:
            index (list[dict]): new list; the entries themselves are shared.
        """
        index = list(index)

        if shuffle_index:
            random.Random(42).shuffle(index)

        if limit is not None:
            index = index[:limit]

        return index
