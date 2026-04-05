from __future__ import annotations

import chess
import torch


BOARD_SIZE = 8

PIECE_CHANNELS = (
    "white_pawn",
    "white_knight",
    "white_bishop",
    "white_rook",
    "white_queen",
    "white_king",
    "black_pawn",
    "black_knight",
    "black_bishop",
    "black_rook",
    "black_queen",
    "black_king",
)

NUM_PIECE_CHANNELS = len(PIECE_CHANNELS)
NUM_STATE_CHANNELS = 6
NUM_CHANNELS = NUM_PIECE_CHANNELS + NUM_STATE_CHANNELS

PIECE_TO_CHANNEL = {
    (chess.WHITE, chess.PAWN): 0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4,
    (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10,
    (chess.BLACK, chess.KING): 11,
}

STATE_CHANNEL_INDEX = {
    "side_to_move": NUM_PIECE_CHANNELS + 0,
    "white_kingside_castling": NUM_PIECE_CHANNELS + 1,
    "white_queenside_castling": NUM_PIECE_CHANNELS + 2,
    "black_kingside_castling": NUM_PIECE_CHANNELS + 3,
    "black_queenside_castling": NUM_PIECE_CHANNELS + 4,
    "en_passant": NUM_PIECE_CHANNELS + 5,
}


def square_to_coords(square: chess.Square) -> tuple[int, int]:
    rank = chess.square_rank(square)
    file_index = chess.square_file(square)
    return 7 - rank, file_index


def board_to_tensor(
    board: chess.Board,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    tensor = torch.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=dtype, device=device)

    for square, piece in board.piece_map().items():
        row, col = square_to_coords(square)
        channel = PIECE_TO_CHANNEL[(piece.color, piece.piece_type)]
        tensor[channel, row, col] = 1.0

    if board.turn == chess.WHITE:
        tensor[STATE_CHANNEL_INDEX["side_to_move"]].fill_(1.0)
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[STATE_CHANNEL_INDEX["white_kingside_castling"]].fill_(1.0)
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[STATE_CHANNEL_INDEX["white_queenside_castling"]].fill_(1.0)
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[STATE_CHANNEL_INDEX["black_kingside_castling"]].fill_(1.0)
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[STATE_CHANNEL_INDEX["black_queenside_castling"]].fill_(1.0)
    if board.ep_square is not None:
        row, col = square_to_coords(board.ep_square)
        tensor[STATE_CHANNEL_INDEX["en_passant"], row, col] = 1.0

    return tensor


def encode_after_move(
    board: chess.Board,
    move: chess.Move,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    next_board = board.copy(stack=False)
    next_board.push(move)
    return board_to_tensor(next_board, device=device, dtype=dtype)
