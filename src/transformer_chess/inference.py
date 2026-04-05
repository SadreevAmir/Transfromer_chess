from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import torch

from .device import get_default_device
from .encoding import board_to_tensor
from .model import BoardValueTransformer, ValueTransformerConfig
from .move_selection import RankedMove, rank_legal_moves


MODEL_CONFIG_KEYS = ("d_model", "depth", "num_heads", "mlp_ratio", "dropout")


@dataclass(frozen=True)
class LoadedModel:
    model: BoardValueTransformer
    device: torch.device
    checkpoint_path: Path
    config: ValueTransformerConfig
    metadata: dict[str, Any]
    value_clip: float


def resolve_device(preferred: str = "auto") -> torch.device:
    preferred = preferred.lower()
    mps_backend = getattr(torch.backends, "mps", None)

    if preferred == "auto":
        return get_default_device()
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA was requested but is not available.")
    if preferred == "mps":
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS was requested but is not available.")
    raise ValueError(f"Unsupported device preference: {preferred}")


def _config_from_payload(payload: dict[str, Any]) -> ValueTransformerConfig:
    source = payload.get("model_config") or payload.get("config")
    if source is None:
        raise KeyError("Checkpoint does not contain model configuration.")

    model_kwargs = {key: source[key] for key in MODEL_CONFIG_KEYS if key in source}
    missing = [key for key in MODEL_CONFIG_KEYS if key not in model_kwargs]
    if missing:
        raise KeyError(f"Checkpoint is missing model config keys: {missing}")
    return ValueTransformerConfig(**model_kwargs)


def load_model_checkpoint(checkpoint_path: str | Path, *, device: str = "auto") -> LoadedModel:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = _config_from_payload(payload)
    target_device = resolve_device(device)
    manifest = payload.get("manifest") or payload.get("dataset_manifest") or {}
    if isinstance(manifest, dict):
        value_clip = float(manifest.get("clip_pawns", payload.get("clip_pawns", 10.0)))
    else:
        value_clip = float(getattr(manifest, "clip_pawns", payload.get("clip_pawns", 10.0)))

    model = BoardValueTransformer(config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(target_device)
    model.eval()
    return LoadedModel(
        model=model,
        device=target_device,
        checkpoint_path=checkpoint_path,
        config=config,
        metadata={key: value for key, value in payload.items() if key != "model_state_dict"},
        value_clip=value_clip,
    )


@torch.no_grad()
def predict_value(loaded_model: LoadedModel, board: chess.Board) -> float:
    inputs = board_to_tensor(board).unsqueeze(0).to(loaded_model.device)
    value = float(loaded_model.model(inputs).item())
    return max(-loaded_model.value_clip, min(loaded_model.value_clip, value))


@torch.no_grad()
def rank_model_moves(loaded_model: LoadedModel, board: chess.Board) -> list[RankedMove]:
    return rank_legal_moves(
        board,
        lambda next_board: predict_value(loaded_model, next_board),
        value_clip=loaded_model.value_clip,
    )


@torch.no_grad()
def select_model_move(loaded_model: LoadedModel, board: chess.Board) -> chess.Move | None:
    ranked = rank_model_moves(loaded_model, board)
    return ranked[0].move if ranked else None
