from __future__ import annotations

import io
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import chess
import zstandard as zstd


LICHESS_EVAL_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"


@dataclass(frozen=True)
class LichessPV:
    move: chess.Move
    score_cp: float
    line_uci: str


@dataclass(frozen=True)
class ValueTrainingPosition:
    fen: str
    best_move: chess.Move
    score_cp: float
    value_target: float
    depth: int
    knodes: int
    pvs: tuple[LichessPV, ...]


def _open_binary_source(source: str | Path) -> BinaryIO:
    source_str = str(source)
    if source_str.startswith(("http://", "https://")):
        return urllib.request.urlopen(source_str)
    return open(source_str, "rb")


def _iter_json_lines_from_binary_stream(stream: BinaryIO, *, compressed: bool) -> Iterator[dict[str, Any]]:
    reader: BinaryIO
    if compressed:
        reader = zstd.ZstdDecompressor().stream_reader(stream)
    else:
        reader = stream

    with io.TextIOWrapper(reader, encoding="utf-8") as text_stream:
        for raw_line in text_stream:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_lichess_eval_json(
    source: str | Path = LICHESS_EVAL_URL,
    *,
    max_positions: int | None = None,
) -> Iterator[dict[str, Any]]:
    source_path = Path(str(source)) if not str(source).startswith(("http://", "https://")) else None
    compressed = source_path.suffix == ".zst" if source_path is not None else str(source).endswith(".zst")

    with _open_binary_source(source) as stream:
        for index, record in enumerate(_iter_json_lines_from_binary_stream(stream, compressed=compressed), start=1):
            yield record
            if max_positions is not None and index >= max_positions:
                break


def _mate_to_centipawns(mate: int, mate_score: int) -> float:
    sign = 1.0 if mate > 0 else -1.0
    return sign * float(mate_score - min(abs(mate), mate_score // 10))


def _pv_score_to_centipawns(pv: dict[str, Any], mate_score: int) -> float:
    if "cp" in pv:
        return float(pv["cp"])
    if "mate" in pv:
        return _mate_to_centipawns(int(pv["mate"]), mate_score)
    raise ValueError("PV entry has neither cp nor mate score.")


def _select_eval_entry(eval_entries: list[dict[str, Any]], top_k: int) -> dict[str, Any] | None:
    if not eval_entries:
        return None

    enough_pvs = [entry for entry in eval_entries if len(entry.get("pvs", [])) >= top_k]
    if enough_pvs:
        return max(
            enough_pvs,
            key=lambda entry: (
                int(entry.get("depth", 0)),
                int(entry.get("knodes", 0)),
                len(entry.get("pvs", [])),
            ),
        )

    candidates = [entry for entry in eval_entries if entry.get("pvs")]
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda entry: (
            len(entry.get("pvs", [])),
            int(entry.get("depth", 0)),
            int(entry.get("knodes", 0)),
        ),
    )


def _orient_scores_to_pv_order(scores: list[float]) -> list[float]:
    if len(scores) < 2:
        return scores

    deltas = sum(next_score - current for current, next_score in zip(scores, scores[1:], strict=False))
    return [-score for score in scores] if deltas > 0 else scores


def clip_centipawn_score(score_cp: float, *, clip_pawns: float) -> float:
    return max(-clip_pawns, min(clip_pawns, score_cp / 100.0))


def record_to_value_position(
    record: dict[str, Any],
    *,
    top_k: int = 4,
    min_pvs: int = 1,
    mate_score: int = 100_000,
    clip_pawns: float = 10.0,
) -> ValueTrainingPosition | None:
    fen = record.get("fen")
    if not fen:
        return None

    board = chess.Board(fen=fen)
    eval_entry = _select_eval_entry(list(record.get("evals", [])), top_k=top_k)
    if eval_entry is None:
        return None

    raw_pvs = list(eval_entry.get("pvs", []))[:top_k]
    if len(raw_pvs) < min_pvs:
        return None

    scores = _orient_scores_to_pv_order([_pv_score_to_centipawns(pv, mate_score) for pv in raw_pvs])
    parsed_pvs: list[LichessPV] = []
    for pv, score in zip(raw_pvs, scores, strict=True):
        line_uci = str(pv.get("line", "")).strip()
        if not line_uci:
            continue
        first_move_uci = line_uci.split()[0]
        move = board.parse_uci(first_move_uci)
        parsed_pvs.append(LichessPV(move=move, score_cp=score, line_uci=line_uci))

    if len(parsed_pvs) < min_pvs:
        return None

    best = parsed_pvs[0]
    return ValueTrainingPosition(
        fen=fen,
        best_move=best.move,
        score_cp=best.score_cp,
        value_target=clip_centipawn_score(best.score_cp, clip_pawns=clip_pawns),
        depth=int(eval_entry.get("depth", 0)),
        knodes=int(eval_entry.get("knodes", 0)),
        pvs=tuple(parsed_pvs),
    )


def iter_value_positions(
    source: str | Path = LICHESS_EVAL_URL,
    *,
    max_positions: int | None = None,
    top_k: int = 4,
    min_pvs: int = 1,
    mate_score: int = 100_000,
    clip_pawns: float = 10.0,
) -> Iterator[ValueTrainingPosition]:
    for record in iter_lichess_eval_json(source, max_positions=max_positions):
        try:
            position = record_to_value_position(
                record,
                top_k=top_k,
                min_pvs=min_pvs,
                mate_score=mate_score,
                clip_pawns=clip_pawns,
            )
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
            continue
        if position is not None:
            yield position


def download_eval_subset(
    output_path: str | Path,
    *,
    source_url: str = LICHESS_EVAL_URL,
    max_positions: int = 1_000_000,
) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for written, record in enumerate(iter_lichess_eval_json(source_url, max_positions=max_positions), start=1):
            handle.write(json.dumps(record, separators=(",", ":")))
            handle.write("\n")
    return written
