#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:12:20 2025

@author: costantino_ai
"""
import re
import chess
from modules import logging

# Function to check if a move is in the list of recommended moves
def check_accuracy(sub_resp, recommended_moves):
    return 1 if ((sub_resp in recommended_moves) or (sub_resp == '' and len(recommended_moves) == 0)) else 0

# Revised function to extract all moves where stockfish_eval is an integer, not just the first
def get_all_moves_with_int_eval(moves, evals):
    moves = eval(moves)
    evals = eval(evals)
    return [move for move, evaluation in zip(moves, evals) if (evaluation is not None) and (int(evaluation) > 0)]

def convert_shorthand_to_long(fen, shorthand_moves, stim_id=999):
    """
    Convert shorthand chess moves into from-cell-to-cell algebraic notation using a given FEN.
    If shorthand_moves is a list, apply conversion to each element of the list.

    Args:
        fen (str): The FEN string representing the current board position.
        shorthand_moves (str or list): The move(s) in shorthand notation (e.g., 'R6e' or ['R6e', 'Nf3']).

    Returns:
        str or list: The move(s) in from-cell-to-cell algebraic notation (e.g., 'a4a6'), or an empty string if conversion is not possible.
    """

    # Check if shorthand_moves is a single shorthand move or a list of moves
    if isinstance(shorthand_moves, list):
        # Apply conversion to each shorthand move in the list
        return [convert_single_shorthand(fen, shorthand_move, stim_id) for shorthand_move in shorthand_moves]
    else:
        # Apply conversion to the single shorthand move
        return convert_single_shorthand(fen, shorthand_moves, stim_id)

def clean_chess_move(move):
    # This pattern will keep only the basic notation for moves, which typically includes
    # the piece symbol (optional, uppercase), followed by the target square (e.g., e4, h5).
    # It removes common annotations like x (capture), + (check), # (checkmate), = (promotion),
    # and any annotations related to check or checkmate, and promotion details.
    cleaned_move = re.sub(r'[x+#=]', '', move)
    return cleaned_move

def convert_single_shorthand(fen, shorthand_move, stim_id):
    """
    Convert a shorthand chess move into from-cell-to-cell algebraic notation using a given FEN.

    Args:
        fen (str): The FEN string representing the current board position.
        shorthand_move (str): The move in shorthand notation (e.g., 'R6e').

    Returns:
        str: The move in from-cell-to-cell algebraic notation (e.g., 'a4a6'), or an empty string if conversion is not possible.
    """
    shorthand_move = clean_chess_move(shorthand_move)

    # Return an empty string immediately if shorthand_move is empty
    if not shorthand_move or shorthand_move == '':
        return ''

    exit_ = False

    # Initialize the board from the provided FEN string.
    board = chess.Board(fen)

    # Attempt to map the first character to a chess piece type.
    piece_type = {
        'R': chess.ROOK,
        'N': chess.KNIGHT,
        'B': chess.BISHOP,
        'Q': chess.QUEEN,
        'K': chess.KING,
        'P': chess.PAWN
    }.get(shorthand_move[0].upper(), None)

    # Validate the piece type and shorthand move format.
    if piece_type is None:
        logging.warning("The piece type shorthand is not recognized.")
        exit_ = True
    if len(shorthand_move) < 3:
        logging.warning("Shorthand move is too short to be valid.")
        exit_ = True

    try:
        # Parse the target square from the shorthand notation.
        target_square = chess.parse_square(shorthand_move[1:].lower())
    except ValueError:
        logging.warning("Shorthand move contains an invalid square.")
        exit_ = True

    if exit_ == False:
        # Search for a legal move that matches the shorthand description.
        for move in board.legal_moves:
            if move.to_square == target_square and board.piece_type_at(move.from_square) == piece_type:
                # Convert the move to 'from-square-to-square' format
                from_square = chess.square_name(move.from_square)
                to_square = chess.square_name(move.to_square)
                return f"{from_square}{to_square}"

        # If we reach here, it means there are no legals moves found
        logging.warning("No legal move found for the provided shorthand notation.")

    # Log a warning and return an empty string if no matching move is found.
    logging.warning(
        f"Returning empty move for the following stimulus: \n\tsub_resp: {shorthand_move}\n\tfen: {fen}\n\tstim_id: {str(int(stim_id))}")
    return ''
