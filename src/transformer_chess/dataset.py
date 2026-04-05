from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterator

import chess
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .encoding import BOARD_SIZE, NUM_CHANNELS, board_to_tensor
from .lichess_eval import LICHESS_EVAL_URL, iter_value_positions


INPUT_BITS = NUM_CHANNELS * BOARD_SIZE * BOARD_SIZE
INPUT_BYTES = INPUT_BITS // 8


@dataclass(frozen=True)
class DatasetShard:
    split: str
    path: str
    num_samples: int


@dataclass(frozen=True)
class DatasetManifest:
    source: str
    max_samples: int | None
    max_source_positions: int | None
    top_k: int
    min_pvs: int
    clip_pawns: float
    shard_size: int
    val_ratio: float
    num_samples: int
    num_train: int
    num_val: int
    skipped: int
    shards: tuple[DatasetShard, ...]


def pack_input_tensor(input_tensor: torch.Tensor) -> np.ndarray:
    flat = input_tensor.to(dtype=torch.uint8).cpu().numpy().reshape(-1)
    return np.packbits(flat, bitorder="little")


def unpack_input_batch(packed_batch: np.ndarray) -> np.ndarray:
    flat = np.unpackbits(packed_batch, count=INPUT_BITS, bitorder="little", axis=-1)
    shape = (packed_batch.shape[0], NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    return flat.reshape(shape).astype(np.float32, copy=False)


def _split_for_fen(fen: str, val_ratio: float) -> str:
    if val_ratio <= 0:
        return "train"
    digest = hashlib.blake2b(fen.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big") / float(1 << 64)
    return "val" if value < val_ratio else "train"


class _ShardBuffer:
    def __init__(self, split: str) -> None:
        self.split = split
        self.inputs_packed: list[np.ndarray] = []
        self.targets: list[np.float16] = []
        self.fens: list[str] = []
        self.best_moves: list[str] = []

    def append(self, *, input_packed: np.ndarray, target: float, fen: str, best_move: str) -> None:
        self.inputs_packed.append(input_packed)
        self.targets.append(np.float16(target))
        self.fens.append(fen)
        self.best_moves.append(best_move)

    def __len__(self) -> int:
        return len(self.fens)

    def clear(self) -> None:
        self.inputs_packed.clear()
        self.targets.clear()
        self.fens.clear()
        self.best_moves.clear()


def _flush_buffer(output_dir: Path, buffer: _ShardBuffer, shard_index: int) -> DatasetShard | None:
    if len(buffer) == 0:
        return None

    split_dir = output_dir / buffer.split
    split_dir.mkdir(parents=True, exist_ok=True)
    shard_path = split_dir / f"shard_{shard_index:05d}.npz"

    np.savez(
        shard_path,
        inputs_packed=np.stack(buffer.inputs_packed, axis=0).astype(np.uint8, copy=False),
        targets=np.asarray(buffer.targets, dtype=np.float16),
        fens=np.asarray(buffer.fens),
        best_moves=np.asarray(buffer.best_moves),
    )
    shard = DatasetShard(split=buffer.split, path=str(shard_path.relative_to(output_dir)), num_samples=len(buffer))
    buffer.clear()
    return shard


def save_manifest(output_dir: str | Path, manifest: DatasetManifest) -> Path:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return manifest_path


def load_manifest(dataset_dir: str | Path) -> DatasetManifest:
    dataset_dir = Path(dataset_dir)
    payload = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    shards = tuple(DatasetShard(**item) for item in payload.pop("shards"))
    allowed = {field.name for field in fields(DatasetManifest) if field.name != "shards"}
    filtered_payload = {key: value for key, value in payload.items() if key in allowed}
    return DatasetManifest(shards=shards, **filtered_payload)


def build_dataset_from_lichess_evals(
    output_dir: str | Path,
    *,
    source: str | Path = LICHESS_EVAL_URL,
    max_samples: int = 1_000_000,
    max_source_positions: int | None = None,
    top_k: int = 4,
    min_pvs: int = 2,
    clip_pawns: float = 10.0,
    shard_size: int = 50_000,
    val_ratio: float = 0.05,
) -> DatasetManifest:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    buffers = {"train": _ShardBuffer("train"), "val": _ShardBuffer("val")}
    shard_indices = {"train": 0, "val": 0}
    shards: list[DatasetShard] = []
    num_samples = 0
    num_train = 0
    num_val = 0
    skipped = 0

    for position in iter_value_positions(
        source,
        max_positions=max_source_positions,
        top_k=top_k,
        min_pvs=min_pvs,
        clip_pawns=clip_pawns,
    ):
        try:
            board = chess.Board(position.fen)
            input_tensor = board_to_tensor(board)
        except Exception:
            skipped += 1
            continue

        split = _split_for_fen(position.fen, val_ratio=val_ratio)
        buffers[split].append(
            input_packed=pack_input_tensor(input_tensor),
            target=position.value_target,
            fen=position.fen,
            best_move=position.best_move.uci(),
        )

        num_samples += 1
        if split == "train":
            num_train += 1
        else:
            num_val += 1

        if len(buffers[split]) >= shard_size:
            shard = _flush_buffer(output_dir, buffers[split], shard_indices[split])
            shard_indices[split] += 1
            if shard is not None:
                shards.append(shard)

        if max_samples is not None and num_samples >= max_samples:
            break

    for split in ("train", "val"):
        shard = _flush_buffer(output_dir, buffers[split], shard_indices[split])
        if shard is not None:
            shards.append(shard)

    manifest = DatasetManifest(
        source=str(source),
        max_samples=max_samples,
        max_source_positions=max_source_positions,
        top_k=top_k,
        min_pvs=min_pvs,
        clip_pawns=clip_pawns,
        shard_size=shard_size,
        val_ratio=val_ratio,
        num_samples=num_samples,
        num_train=num_train,
        num_val=num_val,
        skipped=skipped,
        shards=tuple(shards),
    )
    save_manifest(output_dir, manifest)
    return manifest


class ShardedEvaluationDataset(IterableDataset[tuple[np.ndarray, np.float32]]):
    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        split: str = "train",
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.seed = seed
        self.manifest = load_manifest(self.dataset_dir)
        self.shards = [shard for shard in self.manifest.shards if shard.split == split]

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.float32]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        shards = self.shards[worker_id::num_workers]
        rng = random.Random(self.seed + worker_id)
        if self.shuffle_shards:
            rng.shuffle(shards)

        for shard in shards:
            shard_path = self.dataset_dir / shard.path
            with np.load(shard_path, allow_pickle=False) as arrays:
                inputs_packed = arrays["inputs_packed"]
                targets = arrays["targets"]
                indices = list(range(len(targets)))
                if self.shuffle_within_shard:
                    rng.shuffle(indices)

                for index in indices:
                    yield inputs_packed[index], np.float32(targets[index])


def _collate_packed_samples(samples: list[tuple[np.ndarray, np.float32]]) -> tuple[torch.Tensor, torch.Tensor]:
    packed_inputs, targets = zip(*samples, strict=True)
    packed_batch = np.stack(packed_inputs, axis=0).astype(np.uint8, copy=False)
    target_batch = np.asarray(targets, dtype=np.float32)
    input_batch = torch.from_numpy(unpack_input_batch(packed_batch))
    target_batch_tensor = torch.from_numpy(target_batch)
    return input_batch, target_batch_tensor


def make_dataloader(
    dataset_dir: str | Path,
    *,
    split: str,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 0,
    shuffle: bool = True,
    pin_memory: bool | None = None,
) -> DataLoader:
    dataset = ShardedEvaluationDataset(
        dataset_dir,
        split=split,
        shuffle_shards=shuffle,
        shuffle_within_shard=shuffle,
        seed=seed,
    )
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader_kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": _collate_packed_samples,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)
