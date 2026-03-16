"""Dataset classes for anti-spoofing training and evaluation.

Provides:
- CNSpoofDataset: Standard dataset for evaluation (individual samples).
- MetaDataset: Meta-learning dataset (meta-train/meta-test splits per batch).
"""

import os
import random
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

# Genre to index mapping (consistent across all datasets)
GENRE_TO_IDX = {
    'drama': 0, 'entertainment': 1, 'interview': 2,
    'live_broadcast': 3, 'movie': 4, 'play': 5,
    'recitation': 6, 'singing': 7, 'speech': 8, 'vlog': 9,
}

LABEL_TO_IDX = {'real': 1, 'spoof': 0}


def read_kaldi_file(path: str) -> Dict[str, str]:
    """Read a Kaldi-format file (space-separated key-value pairs).

    Args:
        path: Path to the Kaldi file.

    Returns:
        Dictionary mapping utterance IDs to values.
    """
    mapping = {}
    with open(path, 'r') as f:
        for line in f:
            uttid, content = line.strip().split(maxsplit=1)
            mapping[uttid] = content
    return mapping


class AudioCollator:
    """Handles padding/truncation of variable-length audio batches.

    Args:
        sampling_rate: Audio sampling rate in Hz.
        max_len_sec: Maximum audio length in seconds.
        padding_value: Value used for padding shorter sequences.
    """

    def __init__(self, sampling_rate: int = 16000, max_len_sec: int = 20,
                 padding_value: float = 0.0):
        self.sampling_rate = sampling_rate
        self.max_len_sec = max_len_sec
        self.max_len_samples = sampling_rate * max_len_sec
        self.padding_value = padding_value

    def pad_or_truncate(self, batch: List[torch.Tensor]) -> List[torch.Tensor]:
        """Pad shorter sequences and randomly truncate longer ones."""
        trailing_dims = batch[0].size()[1:]
        max_len = min(
            max(s.size(0) for s in batch),
            self.max_len_samples,
        )

        if all(x.shape[0] == max_len for x in batch):
            return batch

        output = []
        for tensor in batch:
            out = tensor.new_full((max_len,) + trailing_dims, self.padding_value)
            if tensor.size(0) <= max_len:
                out[:tensor.size(0), ...] = tensor
            else:
                start = torch.randint(0, tensor.size(0) - max_len, (1,)).item()
                out[...] = tensor[start:start + max_len, ...]
            output.append(out)
        return output

    def get_effective_length(self, length: int) -> int:
        """Clamp audio length to maximum allowed samples."""
        return min(length, self.max_len_samples)


class CNSpoofDataset(Dataset):
    """Standard anti-spoofing dataset for evaluation.

    Returns individual (audio, label, genre, genre_id, length) tuples.

    Args:
        task: Task name (used to locate data directory).
        split: Data split ('train', 'val', or 'test').
    """

    def __init__(self, task: str, split: str = 'train'):
        data_dir = os.path.join('data', task, split)

        self.utt2wav = read_kaldi_file(os.path.join(data_dir, 'wav.scp'))
        self.utt2label = read_kaldi_file(os.path.join(data_dir, 'utt2spk'))
        self.utt2genre = read_kaldi_file(os.path.join(data_dir, 'utt2genre'))

        self.utterances = list(self.utt2label.keys())

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int):
        uttid = self.utterances[idx]
        data, _ = sf.read(self.utt2wav[uttid])
        audio = torch.from_numpy(data.astype(np.float32)).unsqueeze(1)
        label = LABEL_TO_IDX[self.utt2label[uttid]]
        genre = self.utt2genre[uttid]
        genre_id = GENRE_TO_IDX[genre]
        return audio, label, genre, genre_id, len(data)


class MetaDataset(Dataset):
    """Meta-learning dataset with genre-based meta-train/meta-test splits.

    Each batch is split by genre: most genres go to meta-train,
    one held-out genre goes to meta-test. This encourages learning
    domain-invariant features.

    Args:
        task: Task name (used to locate data directory).
        split: Data split ('train', 'val', or 'test').
    """

    def __init__(self, task: str, split: str = 'train'):
        data_dir = os.path.join('data', task, split)

        self.utt2wav = read_kaldi_file(os.path.join(data_dir, 'wav.scp'))
        self.utt2label = read_kaldi_file(os.path.join(data_dir, 'utt2spk'))
        self.utt2genre = read_kaldi_file(os.path.join(data_dir, 'utt2genre'))

        self.utterances = list(self.utt2label.keys())
        self.genres = sorted(set(self.utt2genre.values()))
        self.genre2idx = {g: i for i, g in enumerate(self.genres)}

        # Build genre → utterance list mapping
        self.genre2uttlist: Dict[str, List[str]] = {}
        for uttid, genre in self.utt2genre.items():
            self.genre2uttlist.setdefault(genre, []).append(uttid)

        self.collator = AudioCollator()

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int):
        # Returns index only; actual data loading happens in collate_fn
        return idx

    @property
    def domain_num(self) -> int:
        """Number of unique genres/domains."""
        return len(self.genres)

    def _load_audio_batch(self, uttids: List[str]) -> Tuple[
        List[torch.Tensor], torch.Tensor, List[int], torch.Tensor
    ]:
        """Load audio, labels, lengths, and genre IDs for utterances.

        Returns:
            audios: List of audio tensors.
            labels: Tensor of binary labels.
            lengths: List of effective audio lengths.
            genre_ids: Tensor of genre indices.
        """
        audios = [
            torch.from_numpy(sf.read(self.utt2wav[u])[0].astype(np.float32)).unsqueeze(1)
            for u in uttids
        ]
        labels = torch.tensor(
            [LABEL_TO_IDX[self.utt2label[u]] for u in uttids],
            dtype=torch.float32,
        )
        lengths = [self.collator.get_effective_length(len(a)) for a in audios]
        genre_ids = torch.tensor(
            [self.genre2idx[self.utt2genre[u]] for u in uttids],
            dtype=torch.int64,
        )
        return audios, labels, lengths, genre_ids

    def _collate_audios(self, audios: List[torch.Tensor]) -> torch.Tensor:
        """Pad/truncate and stack audio tensors into a batch.

        Returns:
            Batch tensor of shape (num_samples, max_len, 1).
        """
        padded = self.collator.pad_or_truncate(audios)
        return (torch.cat(padded, dim=1)
                .permute(1, 0)
                .view(len(padded), -1, 1))

    def random_collate_fn(self, batch: List[int]):
        """Collate: randomly split batch genres into meta-train/meta-test.

        Randomly selects one genre as meta-test; all others are meta-train.
        Retries if meta-test has fewer than 2 samples.

        Returns:
            Tuple of (mtr_audio, mtr_labels, mtr_lengths, mtr_genre_ids,
                      mte_audio, mte_labels, mte_lengths, mte_genre_ids).
        """
        uttids = [self.utterances[idx] for idx in batch]

        # Retry until meta-test has at least 2 samples
        mtr_uttids: List[str] = []
        mte_uttids: List[str] = []
        while len(mte_uttids) < 2:
            batch_genres = list({self.utt2genre[u] for u in uttids})
            assert len(batch_genres) >= 2, "Batch contains only one genre!"

            mtr_genre_set = set(random.sample(batch_genres, k=len(batch_genres) - 1))
            mtr_uttids = [u for u in uttids if self.utt2genre[u] in mtr_genre_set]
            mte_uttids = [u for u in uttids if self.utt2genre[u] not in mtr_genre_set]

        mtr_audios, mtr_labels, mtr_lengths, mtr_gids = self._load_audio_batch(mtr_uttids)
        mte_audios, mte_labels, mte_lengths, mte_gids = self._load_audio_batch(mte_uttids)

        return (
            self._collate_audios(mtr_audios), mtr_labels, mtr_lengths, mtr_gids,
            self._collate_audios(mte_audios), mte_labels, mte_lengths, mte_gids,
        )

    def balance_collate_fn(self, batch: List[int]):
        """Collate: balanced genre sampling for meta-train/meta-test.

        Samples equal numbers per genre. One genre is held out for meta-test.

        Returns:
            Same tuple format as random_collate_fn.
        """
        batch_size = len(batch)
        num_per_genre = batch_size // len(self.genres)

        mtr_genre_names = random.sample(self.genres, k=len(self.genres) - 1)
        mte_genre_names = [g for g in self.genres if g not in mtr_genre_names]

        mtr_uttids = [
            u for genre in mtr_genre_names
            for u in random.sample(self.genre2uttlist[genre], k=num_per_genre)
        ]
        mte_uttids = [
            u for genre in mte_genre_names
            for u in random.sample(self.genre2uttlist[genre], k=num_per_genre)
        ]

        mtr_audios, mtr_labels, mtr_lengths, mtr_gids = self._load_audio_batch(mtr_uttids)
        mte_audios, mte_labels, mte_lengths, mte_gids = self._load_audio_batch(mte_uttids)

        return (
            self._collate_audios(mtr_audios), mtr_labels, mtr_lengths, mtr_gids,
            self._collate_audios(mte_audios), mte_labels, mte_lengths, mte_gids,
        )
