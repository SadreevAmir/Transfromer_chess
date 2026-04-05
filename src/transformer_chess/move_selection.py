from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import chess


@dataclass(frozen=True)
class RankedMove:
    move: chess.Move
    current_player_value: float
    opponent_value: float


def rank_legal_moves(
    board: chess.Board,
    evaluate_after_board: Callable[[chess.Board], float],
    *,
    value_clip: float = 10.0,
) -> list[RankedMove]:
    ranked: list[RankedMove] = []
    for move in board.legal_moves:
        next_board = board.copy(stack=False)
        next_board.push(move)

        if next_board.is_checkmate():
            opponent_value = -value_clip
        elif next_board.is_stalemate() or next_board.is_insufficient_material() or next_board.can_claim_draw():
            opponent_value = 0.0
        else:
            opponent_value = float(evaluate_after_board(next_board))

        ranked.append(
            RankedMove(
                move=move,
                current_player_value=-opponent_value,
                opponent_value=opponent_value,
            )
        )

    return sorted(ranked, key=lambda item: item.current_player_value, reverse=True)


def select_best_move(
    board: chess.Board,
    evaluate_after_board: Callable[[chess.Board], float],
    *,
    value_clip: float = 10.0,
) -> chess.Move | None:
    ranked = rank_legal_moves(board, evaluate_after_board, value_clip=value_clip)
    return ranked[0].move if ranked else None
