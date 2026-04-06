from .cli import build_parser
from .dataset import (
    DatasetManifest,
    DatasetShard,
    ShardedEvaluationDataset,
    build_dataset_from_lichess_evals,
    load_manifest,
    make_dataloader,
    pack_input_tensor,
)
from .device import get_default_device, get_model_device
from .encoding import board_to_tensor, encode_after_move
from .inference import (
    LoadedModel,
    load_model_checkpoint,
    predict_value,
    rank_model_moves,
    resolve_device,
    select_model_move,
)
from .lichess_eval import LICHESS_EVAL_URL, download_eval_subset, iter_value_positions
from .model import BoardValueTransformer, ValueTransformerConfig
from .move_selection import rank_legal_moves, select_best_move
from .train import (
    TRAINING_PRECISION_CHOICES,
    evaluate_loss,
    make_optimizer,
    resolve_training_precision,
    train_epoch,
    train_step,
)

__all__ = [
    "BoardValueTransformer",
    "DatasetManifest",
    "DatasetShard",
    "LICHESS_EVAL_URL",
    "LoadedModel",
    "ShardedEvaluationDataset",
    "TRAINING_PRECISION_CHOICES",
    "ValueTransformerConfig",
    "build_dataset_from_lichess_evals",
    "build_parser",
    "board_to_tensor",
    "download_eval_subset",
    "encode_after_move",
    "evaluate_loss",
    "get_default_device",
    "get_model_device",
    "iter_value_positions",
    "load_manifest",
    "load_model_checkpoint",
    "make_dataloader",
    "make_optimizer",
    "pack_input_tensor",
    "predict_value",
    "rank_legal_moves",
    "rank_model_moves",
    "resolve_training_precision",
    "resolve_device",
    "select_best_move",
    "select_model_move",
    "train_epoch",
    "train_step",
]
