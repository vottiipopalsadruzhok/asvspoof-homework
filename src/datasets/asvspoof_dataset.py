from pathlib import Path

from src.datasets.base_dataset import BaseDataset

# class index of each protocol label; spoof is the negative class
LABEL_MAP = {"spoof": 0, "bonafide": 1}

# partition -> (directory with the flac files, CM protocol file)
PARTITIONS = {
    "train": ("ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt"),
    "dev": ("ASVspoof2019_LA_dev", "ASVspoof2019.LA.cm.dev.trl.txt"),
    "eval": ("ASVspoof2019_LA_eval", "ASVspoof2019.LA.cm.eval.trl.txt"),
}

PROTOCOL_DIR = "ASVspoof2019_LA_cm_protocols"


def read_protocol(protocol_path, audio_dir):
    """
    Build an index out of a CM protocol file. Its lines are

        SPEAKER_ID  UTT_ID  -  ATTACK_ID  LABEL
        LA_0069 LA_D_1105538 - - bonafide
        LA_0052 LA_D_1006888 - A03 spoof

    The speaker is dropped (the countermeasure is speaker independent), the
    attack id is kept to break the scores down by attack.

    Args:
        protocol_path (Path): path to the protocol file.
        audio_dir (Path): directory with the flac files of the partition.
    Returns:
        index (list[dict]): one entry per protocol line, in the same order.
    """
    index = []
    with protocol_path.open() as f:
        for line in f:
            _, utt_id, _, attack_id, label = line.split()
            index.append(
                {
                    "path": str(audio_dir / f"{utt_id}.flac"),
                    "label": LABEL_MAP[label],
                    "utt_id": utt_id,
                    "attack_id": attack_id,
                }
            )
    return index


class ASVSpoofDataset(BaseDataset):
    """
    LA partition of ASVspoof2019.

    The index is rebuilt from the protocol on every run. It takes ~0.25 s for
    the eval partition and, unlike a cached index file, it cannot go stale
    when data_dir changes.
    """

    def __init__(self, part, data_dir, *args, **kwargs):
        """
        Args:
            part (str): "train", "dev" or "eval".
            data_dir (str): directory that holds ASVspoof2019_LA_{part} and
                ASVspoof2019_LA_cm_protocols.
        """
        assert part in PARTITIONS, f"Unknown partition {part}, use {list(PARTITIONS)}"

        audio_subdir, protocol_name = PARTITIONS[part]
        data_dir = Path(data_dir)
        index = read_protocol(
            data_dir / PROTOCOL_DIR / protocol_name, data_dir / audio_subdir / "flac"
        )

        super().__init__(index, *args, **kwargs)
