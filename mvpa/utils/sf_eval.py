import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import pdist, squareform
import chess
import chess.engine
import logging
from multiprocessing import Pool, cpu_count

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration ---
from config import MISC_PATH
CSV_PATH = os.path.join(str(MISC_PATH), "fmri-ds_40-stims.csv")
IMAGE_FOLDER = "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-dataset-vis/data/stimuli"
STOCKFISH_PATH = "/home/eik-tb/Documents/misctools/stockfish-ubuntu-x86-64/stockfish/stockfish-ubuntu-x86-64"
IMAGE_SIZE = (256, 256)
TIME_LIMIT = 0.1
PUNISH_THRESHOLD = 200

# --- Utility Functions ---

def evaluate_expectation(fen, engine):
    board = chess.Board(fen)
    info = engine.analyse(board, chess.engine.Limit(time=TIME_LIMIT))
    score = info["score"]
    if score.is_mate():
        mate = score.relative.mate()
        return max(0.0, min(1.0, 1.0 - 0.0 * (mate - 1) if mate > 0 else 0.0 + 0.0 * (abs(mate) - 1)))
    return score.relative.wdl(model="sf").expectation()

def compute_visual_rdm(filenames):
    images = []
    for fname in filenames:
        path = os.path.join(IMAGE_FOLDER, fname)
        img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
        images.append(np.array(img).astype(np.float32).flatten())
    return squareform(pdist(np.vstack(images), metric="correlation"))

def compute_tactical_pressure(fen):
    board = chess.Board(fen)
    feats = {}
    piece_types = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook", chess.QUEEN: "queen"}
    for color, label in [(chess.WHITE, "white"), (chess.BLACK, "black")]:
        attacks, pinned, hanging = {k: 0 for k in piece_types.values()}, 0, 0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p and p.color == color:
                pt = piece_types.get(p.piece_type)
                if pt: attacks[pt] += len(board.attacks(sq))
                if not board.is_attacked_by(not color, sq) and any(board.is_attacked_by(color, t) for t in board.attacks(sq)):
                    hanging += 1
                if board.is_pinned(color, sq): pinned += 1
        feats.update({f"{label}_{k}_pressure": v for k, v in attacks.items()})
        feats[f"{label}_pinned"] = pinned
        feats[f"{label}_hanging"] = hanging
    return feats

def detect_motifs(fen):
    board = chess.Board(fen)
    motifs = {
        "back_rank_mate": False, "smothered_mate": False, "fork": False, "pin": False,
        "skewer": False, "discovered_attack": False, "double_check": False,
        "isolated_pawn": False, "doubled_pawn": False, "passed_pawn": False
    }
    for move in board.legal_moves:
        if board.gives_check(move):
            motifs["double_check"] |= True
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.KNIGHT:
            motifs["fork"] = True
        if board.is_pinned(board.turn, move.from_square):
            motifs["pin"] = True
    for file in range(8):
        pawns = [sq for sq in chess.SquareSet(chess.BB_FILES[file]) if board.piece_at(sq) == chess.Piece(chess.PAWN, chess.WHITE)]
        if len(pawns) >= 2:
            motifs["doubled_pawn"] = True
    return motifs

def compute_punishability(fen, engine, threshold=PUNISH_THRESHOLD):
    board = chess.Board(fen)
    base_eval = engine.analyse(board, chess.engine.Limit(time=TIME_LIMIT))["score"].white().score(mate_score=100000)
    bad_moves = 0
    for move in board.legal_moves:
        board.push(move)
        try:
            new_eval = engine.analyse(board, chess.engine.Limit(time=TIME_LIMIT))["score"].white().score(mate_score=100000)
            if base_eval is not None and new_eval is not None and base_eval - new_eval > threshold:
                bad_moves += 1
        except Exception:
            pass
        board.pop()
    return bad_moves

def compute_material_imbalance(fen):
    board = chess.Board(fen)
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    white_score = sum(piece_values.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.WHITE)
    black_score = sum(piece_values.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.BLACK)
    return white_score - black_score

def compute_king_safety(fen):
    board = chess.Board(fen)
    def count_threats(king_square, attacker_color):
        ring = chess.SquareSet(chess.BB_KING_ATTACKS[king_square])
        return sum(1 for sq in ring if board.is_attacked_by(attacker_color, sq))
    wk_sq = board.king(chess.WHITE)
    bk_sq = board.king(chess.BLACK)
    return count_threats(wk_sq, chess.BLACK), count_threats(bk_sq, chess.WHITE)

def plot_rdm(matrix, title, label):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap="magma", xticklabels=False, yticklabels=False, cbar_kws={"label": label})
    plt.title(title)
    plt.tight_layout()
    plt.show()

def extract_features(args):
    i, fen, filename, stim_id = args
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        engine.configure({"UCI_ShowWDL": True})
        white_safety, black_safety = compute_king_safety(fen)
        return {
            "stim_id": stim_id,
            "filename": filename,
            "fen": fen,
            "win_prob": evaluate_expectation(fen, engine),
            "punishability": compute_punishability(fen, engine),
            "material_imbalance": compute_material_imbalance(fen),
            "white_king_safety": white_safety,
            "black_king_safety": black_safety,
            **compute_tactical_pressure(fen),
            **detect_motifs(fen)
        }

# --- Load Stimuli ---
logging.info("Loading stimuli CSV...")
df = pd.read_csv(CSV_PATH)
fens = df["fen"].tolist()
filenames = df["filename"].tolist()
logging.info(f"Loaded {len(fens)} positions.")

# --- Feature Extraction ---
records = []
with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
    engine.configure({"UCI_ShowWDL": True})
    logging.info("Starting parallel feature extraction...")
    args_list = list(zip(range(len(fens)), fens, filenames, df["stim_id"].tolist()))
    with Pool(min(cpu_count(), 12)) as pool:
        records = list(pool.map(extract_features, args_list))
    logging.info("Feature extraction complete.")


feature_df = pd.DataFrame(records)

# --- RDMs ---
def compute_and_plot_rdm(dataframe, columns, title):
    logging.info(f"Computing RDM for: {title}")
    data = dataframe[columns].values
    rdm = squareform(pdist(data, metric="euclidean"))
    plot_rdm(rdm, title, "Euclidean Distance")

compute_and_plot_rdm(feature_df, ["win_prob"], "RDM - Win Expectation")
compute_and_plot_rdm(feature_df, ["punishability"], "RDM - Punishability")
compute_and_plot_rdm(feature_df, ["material_imbalance"], "RDM - Material Imbalance")
compute_and_plot_rdm(feature_df, ["white_king_safety", "black_king_safety"], "RDM - King Safety")
compute_and_plot_rdm(feature_df, [c for c in feature_df.columns if "_pressure" in c or "_pinned" in c or "_hanging" in c], "RDM - Tactical Pressure")
compute_and_plot_rdm(feature_df, [c for c in feature_df.columns if c in ["back_rank_mate", "smothered_mate", "fork", "pin", "skewer", "discovered_attack", "double_check", "isolated_pawn", "doubled_pawn", "passed_pawn"]], "RDM - Tactical Motifs")
plot_rdm(compute_visual_rdm(feature_df["filename"]), "RDM - RGB Pixel Distance", "Correlation Distance")
