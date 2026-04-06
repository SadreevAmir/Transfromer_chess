from __future__ import annotations

import argparse
from pathlib import Path

import chess
import torch

from .dataset import build_dataset_from_lichess_evals, load_manifest, make_dataloader
from .device import get_default_device
from .inference import load_model_checkpoint, predict_value, rank_model_moves
from .lichess_eval import LICHESS_EVAL_URL, download_eval_subset
from .model import BoardValueTransformer, ValueTransformerConfig
from .train import (
    TRAINING_PRECISION_CHOICES,
    evaluate_loss,
    make_optimizer,
    resolve_training_precision,
    train_epoch,
)


def _add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)


def _model_from_args(args: argparse.Namespace) -> BoardValueTransformer:
    config = ValueTransformerConfig(
        d_model=args.d_model,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    )
    return BoardValueTransformer(config)


def _download_command(args: argparse.Namespace) -> int:
    written = download_eval_subset(
        args.output,
        source_url=args.source_url,
        max_positions=args.max_positions,
    )
    print(f"saved {written} positions to {args.output}")
    return 0


def _build_dataset_command(args: argparse.Namespace) -> int:
    manifest = build_dataset_from_lichess_evals(
        args.output_dir,
        source=args.input,
        max_samples=args.max_samples,
        max_source_positions=args.max_source_positions,
        top_k=args.top_k,
        min_pvs=args.min_pvs,
        clip_pawns=args.clip_pawns,
        shard_size=args.shard_size,
        val_ratio=args.val_ratio,
    )
    print(
        f"built dataset at {args.output_dir} "
        f"(samples={manifest.num_samples}, train={manifest.num_train}, val={manifest.num_val}, skipped={manifest.skipped})"
    )
    return 0


def _train_command(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir)
    manifest = load_manifest(dataset_dir)
    if manifest.num_train == 0:
        raise ValueError("Dataset has no training samples.")

    device = get_default_device()
    precision = resolve_training_precision(device, args.precision)
    model = _model_from_args(args)
    optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    print(f"training device={device} precision={precision}")

    for epoch in range(args.epochs):
        train_loader = make_dataloader(
            dataset_dir,
            split="train",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed + epoch,
            shuffle=True,
        )
        train_loss = train_epoch(
            model,
            optimizer,
            train_loader,
            device=device,
            precision=precision,
            max_batches=args.max_train_batches,
        )

        summary = f"epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.6f}"
        if manifest.num_val > 0:
            val_loader = make_dataloader(
                dataset_dir,
                split="val",
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
                shuffle=False,
            )
            val_loss = evaluate_loss(
                model,
                val_loader,
                device=device,
                precision=precision,
                max_batches=args.max_val_batches,
            )
            summary += f", val_loss={val_loss:.6f}"
        print(summary)

    checkpoint_path = Path(args.output)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "d_model": args.d_model,
                "depth": args.depth,
                "num_heads": args.num_heads,
                "mlp_ratio": args.mlp_ratio,
                "dropout": args.dropout,
            },
            "training_config": vars(args),
            "manifest": manifest,
            "clip_pawns": manifest.clip_pawns,
        },
        checkpoint_path,
    )
    print(f"saved checkpoint to {checkpoint_path}")
    return 0


def _predict_value_command(args: argparse.Namespace) -> int:
    loaded = load_model_checkpoint(args.checkpoint, device=args.device)
    board = chess.Board(args.fen)
    value = predict_value(loaded, board)
    print(f"value={value:.4f}")
    return 0


def _select_move_command(args: argparse.Namespace) -> int:
    loaded = load_model_checkpoint(args.checkpoint, device=args.device)
    board = chess.Board(args.fen)
    ranked = rank_model_moves(loaded, board)
    if not ranked:
        print("no legal moves")
        return 0
    for item in ranked[: args.top_n]:
        print(
            f"{board.san(item.move):<8} uci={item.move.uci()} "
            f"current_value={item.current_player_value:.4f} opponent_value={item.opponent_value:.4f}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transformer-chess")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download-lichess-evals")
    download_parser.add_argument("--source-url", default=LICHESS_EVAL_URL)
    download_parser.add_argument("--output", required=True)
    download_parser.add_argument("--max-positions", type=int, default=1_000_000)
    download_parser.set_defaults(func=_download_command)

    build_parser = subparsers.add_parser("build-dataset")
    build_parser.add_argument("--input", default=LICHESS_EVAL_URL)
    build_parser.add_argument("--output-dir", required=True)
    build_parser.add_argument("--max-samples", type=int, default=1_000_000)
    build_parser.add_argument("--max-source-positions", type=int, default=None)
    build_parser.add_argument("--top-k", type=int, default=4)
    build_parser.add_argument("--min-pvs", type=int, default=2)
    build_parser.add_argument("--clip-pawns", type=float, default=10.0)
    build_parser.add_argument("--shard-size", type=int, default=50_000)
    build_parser.add_argument("--val-ratio", type=float, default=0.05)
    build_parser.set_defaults(func=_build_dataset_command)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset-dir", required=True)
    train_parser.add_argument("--output", default="artifacts/value_model.pt")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--max-train-batches", type=int, default=None)
    train_parser.add_argument("--max-val-batches", type=int, default=None)
    train_parser.add_argument("--precision", default="auto", choices=TRAINING_PRECISION_CHOICES)
    _add_common_model_args(train_parser)
    train_parser.set_defaults(func=_train_command)

    predict_parser = subparsers.add_parser("predict-value")
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--fen", required=True)
    predict_parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    predict_parser.set_defaults(func=_predict_value_command)

    select_parser = subparsers.add_parser("select-move")
    select_parser.add_argument("--checkpoint", required=True)
    select_parser.add_argument("--fen", required=True)
    select_parser.add_argument("--top-n", type=int, default=5)
    select_parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    select_parser.set_defaults(func=_select_move_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
