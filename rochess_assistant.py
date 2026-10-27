"""
Usage (see README.md for the full walkthrough):
    py rochess_assistant.py --image image.png        # test on a screenshot
    py rochess_assistant.py --calibrate              # find the board on screen once
    py rochess_assistant.py                          # live mode with overlay
    py rochess_assistant.py --debug                  # live mode + debug window

The file is organised top to bottom as:
    1. SETTINGS            - everything you are likely to tweak
    2. Board geometry      - finding the board and cutting it into 64 squares
    3. Piece recognition   - "is there a piece?" and "which piece is it?"
    4. Position tracking   - turning the 64 labels into a legal chess.Board
    5. Engine              - Stockfish + plain-English move descriptions
    6. Overlay & debug     - drawing on screen
    7. Modes               - --image, --calibrate, live loop
"""

import argparse
import ctypes
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import chess
import chess.engine

# =============================================================================
# 1. SETTINGS - change these to suit your setup
# =============================================================================

# Which side you play: "white", "black", or "auto" (auto = whichever colour's
# king starts at the bottom of the screen is you).
PLAY_AS = "auto"

# Full path to the Stockfish program you downloaded (see README.md).
STOCKFISH_PATH = r"C:\Users\cyubi\Downloads\stockfish-windows-x86-64-universal\stockfish\stockfish-windows-x86-64-universal.exe"
THINK_TIME = 0.5            # seconds Stockfish thinks per move

# Where the board is on screen, as (left, top, width, height) in pixels.
# Leave as None to use calibration.json (made by --calibrate) or auto-detection.
BOARD_REGION = None
CALIBRATION_FILE = "calibration.json"
MONITOR_INDEX = 0           # 0 = all monitors together, 1 = first, 2 = second...

# Live loop
LOOP_DELAY_MS = 250         # time between screen captures
STABLE_FRAMES = 2           # board must look the same this many captures in a row
                            # before we react (avoids reading pieces mid-animation)
DEBUG = False               # same as passing --debug

# --- Board auto-detection ----------------------------------------------------
# The dark (olive green) squares in OpenCV HSV units (H 0-179, S/V 0-255).
# Measured from image.png: dark square = H 45, S 115, V 149.
DARK_SQUARE_HSV_LOW = (35, 70, 90)
DARK_SQUARE_HSV_HIGH = (55, 170, 210)

# --- Piece recognition -------------------------------------------------------
TEMPLATE_DIR = "templates"
SQUARE_SIZE = 64            # every square is resized to this many pixels
BG_TOLERANCE = 28           # colour distance from the square colour that counts
                            # as "part of a piece" (lower = more sensitive)
OUTLINE_MAX_GRAY = 70       # pixels darker than this are the black piece outline
MIN_OUTLINE_FRACTION = 0.03 # occupied square needs at least 3% outline pixels...
MIN_PIECE_FRACTION = 0.10   # ...and at least 10% non-background pixels
OCCUPANCY_MARGIN = 0.15     # ignore this fraction at each edge when testing
                            # occupancy (keeps the a-h / 1-8 labels out)
FEATURE_MARGIN = 0.06       # edge trimmed before template matching
MATCH_SHIFT = 3             # pixels a piece may be off-centre vs its template
MATCH_THRESHOLD = 0.55      # best template score below this = "unsure" square
SAME_SHADE_BONUS = 0.03     # small preference for templates from the same
                            # square colour (light/dark)
BACKGROUND_FILL = 160       # grey value that replaces the square colour

# --- Overlay -----------------------------------------------------------------
SHOW_OVERLAY = True
ARROW_COLOR = "#00b4ff"
ARROW_WIDTH = 10
HIGHLIGHT_COLOR = "#ffcc00"
HIGHLIGHT_WIDTH = 5

# =============================================================================
# Small helpers
# =============================================================================

PIECE_LABELS = [c + t for c in "wb" for t in "PNBRQK"]   # "wP", "bK", ...


def label_to_piece(label):
    """'wN' -> chess.Piece(KNIGHT, WHITE)"""
    return chess.Piece.from_symbol(label[1] if label[0] == "w" else label[1].lower())


def piece_to_label(piece):
    """chess.Piece -> 'wN' / 'bp' style label used for template file names."""
    return ("w" if piece.color == chess.WHITE else "b") + piece.symbol().upper()


def make_dpi_aware():
    """On Windows with display scaling (125%, 150%...) screen coordinates get
    scaled unless we declare ourselves DPI aware. Without this the capture and
    the overlay would not line up with the real board."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


# =============================================================================
# 2. Board geometry
# =============================================================================

def detect_board(img):
    """Find the 8x8 board in a screenshot. Returns (x, y, w, h) or None.

    Idea: the 32 dark squares are all the same olive green. We make a mask of
    that colour, find the square-shaped blobs of equal size, and the box
    around all of them is exactly the board (dark squares touch every edge
    row and column). Profile pictures, timers etc. are ignored because they
    are not equal-sized green squares."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, DARK_SQUARE_HSV_LOW, DARK_SQUARE_HSV_HIGH)
    # Dark squares touch at their corners; shrinking the mask a little keeps
    # each square as its own blob. We add the shrink back below.
    erode = 2
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=erode)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x, y, w, h = x - erode, y - erode, w + 2 * erode, h + 2 * erode
        if w >= 12 and h >= 12 and 0.8 < w / h < 1.25:
            rects.append((x, y, w, h))
    if len(rects) < 16:
        return None

    sizes = np.array([(w + h) / 2 for _, _, w, h in rects])
    median = np.median(sizes)
    squares = [r for r, s in zip(rects, sizes) if abs(s - median) < 0.15 * median]
    if len(squares) < 16:
        return None

    x0 = min(r[0] for r in squares)
    y0 = min(r[1] for r in squares)
    x1 = max(r[0] + r[2] for r in squares)
    y1 = max(r[1] + r[3] for r in squares)
    w, h = x1 - x0, y1 - y0
    # Sanity check: the result should be roughly 8 squares wide and tall.
    if not (6.5 < w / median < 9.5 and 6.5 < h / median < 9.5):
        return None
    return (int(x0), int(y0), int(w), int(h))


def click_region(img, window="Click the board corners", hint=None):
    """Let the user click the top-left and bottom-right corners of the board.
    Press Enter to accept the yellow 'hint' rectangle instead (if shown).
    Returns (x, y, w, h) in the image's own pixel coordinates, or None."""
    scale = min(1.0, 1400 / img.shape[1], 850 / img.shape[0])
    view = cv2.resize(img, None, fx=scale, fy=scale)
    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            clicks.append((x / scale, y / scale))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        canvas = view.copy()
        msg = "Click TOP-LEFT corner of the board, then BOTTOM-RIGHT. Esc = cancel"
        if hint:
            hx, hy, hw, hh = [int(v * scale) for v in hint]
            cv2.rectangle(canvas, (hx, hy), (hx + hw, hy + hh), (0, 255, 255), 2)
            msg = "Enter = accept yellow box, or click TOP-LEFT then BOTTOM-RIGHT. Esc = cancel"
        for cx, cy in clicks:
            cv2.circle(canvas, (int(cx * scale), int(cy * scale)), 6, (0, 0, 255), -1)
        cv2.putText(canvas, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(canvas, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if len(clicks) == 2:
            break
        if key == 27:
            cv2.destroyWindow(window)
            return None
        if key == 13 and hint:
            cv2.destroyWindow(window)
            return hint
    cv2.destroyWindow(window)
    (x0, y0), (x1, y1) = clicks
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return (int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0)))


def crop(img, region):
    x, y, w, h = region
    return img[y:y + h, x:x + w]


def split_squares(board_img):
    """Cut the board image into an 8x8 list of square images.
    squares[row][col], row 0 = top of the screen, col 0 = left."""
    h, w = board_img.shape[:2]
    xs = np.linspace(0, w, 9).round().astype(int)
    ys = np.linspace(0, h, 9).round().astype(int)
    return [[board_img[ys[r]:ys[r + 1], xs[c]:xs[c + 1]] for c in range(8)]
            for r in range(8)]


def screen_to_square(row, col, white_at_bottom):
    """Screen position -> chess square. When you play black the board is
    flipped, so the top-left square is h1 instead of a8."""
    if white_at_bottom:
        return chess.square(col, 7 - row)
    return chess.square(7 - col, row)


def square_to_screen(square, white_at_bottom):
    f, r = chess.square_file(square), chess.square_rank(square)
    if white_at_bottom:
        return 7 - r, f          # (row, col)
    return r, 7 - f


def square_shade(row, col):
    """The top-left square is light in both orientations (a8 and h1 are light)."""
    return "light" if (row + col) % 2 == 0 else "dark"


# =============================================================================
# 3. Piece recognition
# =============================================================================
#
# Why template matching instead of a trained classifier?
#   RoChess draws every piece from the same fixed cartoon artwork, so a white
#   knight always looks pixel-for-pixel the same apart from the square colour
#   behind it. A trained classifier would need hundreds of labelled images to
#   learn something we can get from one screenshot. Template matching needs a
#   single example per piece, is easy to inspect (the templates are just
#   PNG files) and easy to fix (add one more PNG).
#
# Handling light vs dark squares:
#   1. Before matching, every pixel that looks like the square colour is
#      replaced by one neutral grey (BACKGROUND_FILL). A knight on a light
#      square and a knight on a dark square then look nearly identical.
#   2. Templates remember the square colour they came from, and we give a
#      small bonus to templates from the same shade.
#
# Recognition happens in two steps per square:
#   a) Occupancy: is there a piece? (needs black outline pixels + enough
#      non-background pixels). Empty squares never reach step b.
#   b) Classification: compare against every template, best score wins.

def board_palette(squares):
    """Median colour of the light and dark squares, measured from the edges
    of every square. Measuring it live means move highlights or a different
    board theme don't need new constants."""
    light, dark = [], []
    for r in range(8):
        for c in range(8):
            sq = cv2.resize(squares[r][c], (SQUARE_SIZE, SQUARE_SIZE))
            ring = np.concatenate([sq[:3].reshape(-1, 3), sq[-3:].reshape(-1, 3),
                                   sq[:, :3].reshape(-1, 3), sq[:, -3:].reshape(-1, 3)])
            (light if square_shade(r, c) == "light" else dark).append(np.median(ring, axis=0))
    return np.median(light, axis=0), np.median(dark, axis=0)


def piece_mask(sq, palette):
    """True where a pixel belongs to a piece: different from this square's own
    background colour AND from both board colours (the latter removes the
    a-h / 1-8 coordinate labels, which are drawn in the other square colour)."""
    ring = np.concatenate([sq[:3].reshape(-1, 3), sq[-3:].reshape(-1, 3),
                           sq[:, :3].reshape(-1, 3), sq[:, -3:].reshape(-1, 3)])
    own_bg = np.median(ring, axis=0)
    px = sq.astype(np.float32)
    mask = np.linalg.norm(px - own_bg, axis=2) > BG_TOLERANCE
    for colour in palette:
        mask &= np.linalg.norm(px - colour, axis=2) > BG_TOLERANCE
    return mask


def occupancy_stats(sq, palette):
    """Return (outline_fraction, piece_fraction) for the centre of a square."""
    m = int(SQUARE_SIZE * OCCUPANCY_MARGIN)
    gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)[m:-m, m:-m]
    mask = piece_mask(sq, palette)[m:-m, m:-m]
    return float((gray < OUTLINE_MAX_GRAY).mean()), float(mask.mean())


def is_occupied(stats):
    outline, piece = stats
    return outline >= MIN_OUTLINE_FRACTION and piece >= MIN_PIECE_FRACTION


def square_feature(sq, palette):
    """Greyscale image of the piece with the square colour replaced by a
    neutral grey. This is what gets compared with the templates."""
    gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY).astype(np.float32)
    feat = np.where(piece_mask(sq, palette), gray, np.float32(BACKGROUND_FILL))
    m = int(SQUARE_SIZE * FEATURE_MARGIN)
    return feat[m:-m, m:-m]


def normalise_square(square_img):
    return cv2.resize(square_img, (SQUARE_SIZE, SQUARE_SIZE), interpolation=cv2.INTER_AREA)


def load_templates(template_dir=TEMPLATE_DIR):
    """Load templates/<label>_<shade>_<anything>.png, e.g. wN_dark_image_g1.png.
    Each template is a raw square crop; we store the palette it was saved with
    in the file name's sibling JSON so its feature is computed the same way."""
    templates = []
    for path in sorted(glob.glob(os.path.join(template_dir, "*.png"))):
        parts = os.path.basename(path).split("_")
        if len(parts) < 2 or parts[0] not in PIECE_LABELS or parts[1] not in ("light", "dark"):
            print(f"[templates] skipping badly named file: {path}")
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        meta_path = path[:-4] + ".json"
        try:
            with open(meta_path) as f:
                palette = [np.array(c, dtype=np.float32) for c in json.load(f)["palette"]]
        except (OSError, ValueError, KeyError):
            palette = [np.array([208, 236, 235], np.float32), np.array([82, 149, 115], np.float32)]
        feat = square_feature(normalise_square(img), palette)
        s = MATCH_SHIFT
        templates.append({"label": parts[0], "shade": parts[1], "path": path,
                          "feature": np.ascontiguousarray(feat[s:-s, s:-s])})
    return templates


def save_template(square_img, label, shade, palette, name, template_dir=TEMPLATE_DIR):
    """Save a square crop as a template (+ the board colours it was seen with)."""
    os.makedirs(template_dir, exist_ok=True)
    base = os.path.join(template_dir, f"{label}_{shade}_{name}")
    cv2.imwrite(base + ".png", square_img)
    with open(base + ".json", "w") as f:
        json.dump({"palette": [list(map(float, c)) for c in palette]}, f)
    return base + ".png"


def classify(feat, shade, templates):
    """Compare one square's feature with every template.
    Returns (best_label, best_score). Scores are -1..1 (1 = identical)."""
    best_label, best_score = None, -2.0
    for t in templates:
        res = cv2.matchTemplate(feat, t["feature"], cv2.TM_CCOEFF_NORMED)
        score = float(np.nan_to_num(res).max())
        if t["shade"] == shade:
            score += SAME_SHADE_BONUS
        if score > best_score:
            best_label, best_score = t["label"], score
    return best_label, best_score


def recognize_board(board_img, templates):
    """Recognise all 64 squares. Returns a list of 8 rows of 8 dicts:
        {"label": "wN" or None, "score": float, "stats": (outline, piece)}"""
    squares = split_squares(board_img)
    palette = board_palette(squares)
    grid = []
    for r in range(8):
        row = []
        for c in range(8):
            sq = normalise_square(squares[r][c])
            stats = occupancy_stats(sq, palette)
            cell = {"label": None, "score": 0.0, "stats": stats}
            if is_occupied(stats):
                cell["label"], cell["score"] = classify(square_feature(sq, palette),
                                                        square_shade(r, c), templates)
            row.append(cell)
        grid.append(row)
    return grid, palette


def grid_key(grid):
    return tuple(cell["label"] for row in grid for cell in row)


def white_is_at_bottom(grid):
    if PLAY_AS == "white":
        return True
    if PLAY_AS == "black":
        return False
    # auto: find the white king
    for r in range(8):
        for c in range(8):
            if grid[r][c]["label"] == "wK":
                return r >= 4
    return True


def unsure_squares(grid, white_at_bottom):
    out = []
    for r in range(8):
        for c in range(8):
            cell = grid[r][c]
            if cell["label"] and cell["score"] < MATCH_THRESHOLD:
                out.append((chess.square_name(screen_to_square(r, c, white_at_bottom)), cell))
    return out


def grid_to_piece_map(grid, white_at_bottom):
    pm = {}
    for r in range(8):
        for c in range(8):
            if grid[r][c]["label"]:
                pm[screen_to_square(r, c, white_at_bottom)] = label_to_piece(grid[r][c]["label"])
    return pm


# =============================================================================
# 4. Position tracking
# =============================================================================
#
# A FEN needs more than piece placement: whose turn it is, castling rights and
# en-passant. A screenshot can't tell us those directly, so we keep a real
# chess.Board and, when the screen changes, look for the legal move (or pair of
# moves - bots reply fast) that turns the old position into the new one.
# If nothing matches (program started mid-game, a frame was misread...) we
# "resync": build a fresh board from the placement and make sensible guesses.

STARTING_MAP = chess.Board().piece_map()


def build_fresh_board(piece_map, my_color, forced_turn=None):
    b = chess.Board(None)
    b.set_piece_map(piece_map)
    # Allow castling only where king and rook still stand on their home squares.
    b.set_castling_fen("KQkq")
    b.castling_rights = b.clean_castling_rights()
    if forced_turn is not None:
        b.turn = forced_turn
        return (b, None) if b.is_valid() else (None, b.status())
    b.turn = chess.WHITE if piece_map == STARTING_MAP else my_color
    if not b.is_valid():
        b.turn = not b.turn          # e.g. the opponent is in check -> their turn
        if not b.is_valid():
            b.turn = not b.turn
            return None, b.status()
    return b, None


class GameTracker:
    def __init__(self):
        self.board = None

    def update(self, piece_map, my_color):
        """Returns (board, description, problem)."""
        if self.board is not None:
            if self.board.piece_map() == piece_map:
                return self.board, "no change", None
            moves = self._find_moves(piece_map)
            if moves:
                sans = []
                for m in moves:
                    sans.append(self.board.san(m))
                    self.board.push(m)
                return self.board, "played: " + " ".join(sans), None
        board, problem = build_fresh_board(piece_map, my_color)
        self.board = board
        return board, "resynced from screen", problem

    def _find_moves(self, piece_map):
        b = self.board
        for m1 in list(b.legal_moves):
            b.push(m1)
            if b.piece_map() == piece_map:
                b.pop()
                return [m1]
            for m2 in list(b.legal_moves):
                b.push(m2)
                hit = b.piece_map() == piece_map
                b.pop()
                if hit:
                    b.pop()
                    return [m1, m2]
            b.pop()
        return None


# =============================================================================
# 5. Engine
# =============================================================================

def open_engine():
    if not os.path.isfile(STOCKFISH_PATH):
        sys.exit(f"Stockfish not found at:\n  {STOCKFISH_PATH}\n"
                 "Download it from https://stockfishchess.org/download/ and set "
                 "STOCKFISH_PATH at the top of rochess_assistant.py.")
    return chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)


def best_move(engine, board):
    info = engine.analyse(board, chess.engine.Limit(time=THINK_TIME))
    return info["pv"][0], info["score"].pov(board.turn)


def describe_move(board, move):
    """Turn a move into a sentence like 'Move the knight from g1 to f3'."""
    piece = board.piece_at(move.from_square)
    frm, to = chess.square_name(move.from_square), chess.square_name(move.to_square)
    if board.is_castling(move):
        side = "kingside" if chess.square_file(move.to_square) > 4 else "queenside"
        text = f"Castle {side} (move the king from {frm} to {to})"
    else:
        text = f"Move the {chess.piece_name(piece.piece_type)} from {frm} to {to}"
        if board.is_en_passant(move):
            text += ", capturing the pawn en passant"
        elif board.is_capture(move):
            text += f", capturing the {chess.piece_name(board.piece_at(move.to_square).piece_type)}"
        if move.promotion:
            text += f", and promote to a {chess.piece_name(move.promotion)}"
    if board.gives_check(move):
        text += " (check!)"
    return f"{text}   [{board.san(move)}]"


def describe_score(score):
    if score.is_mate():
        n = score.mate()
        return f"you mate in {n}" if n > 0 else f"opponent mates in {-n}"
    cp = score.score() / 100
    return f"{cp:+.2f} (positive = good for you)"


# =============================================================================
# 6. Overlay & debug drawing
# =============================================================================

class Overlay:
    """A borderless, always-on-top, see-through, click-through window placed
    exactly over the board. Everything painted in TRANSPARENT colour is
    invisible, so only the arrow and highlights show up."""
    TRANSPARENT = "#ff00fe"

    def __init__(self, root, region):
        import tkinter as tk
        self.tk = tk
        self.region = region
        x, y, w, h = region
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)            # no title bar / border
        self.win.geometry(f"{w}x{h}+{x}+{y}")
        self.win.attributes("-topmost", True)
        self.win.config(bg=self.TRANSPARENT)
        self.win.attributes("-transparentcolor", self.TRANSPARENT)
        self.canvas = tk.Canvas(self.win, width=w, height=h, bg=self.TRANSPARENT,
                                highlightthickness=0)
        self.canvas.pack()
        self.win.update_idletasks()
        self.hidden_from_capture = self._setup_window_styles()

    def _setup_window_styles(self):
        """Use the Windows API to (1) let mouse clicks pass through to Roblox and
        (2) hide the overlay from screen capture, so our own arrow never
        confuses the piece recognition."""
        user32 = ctypes.windll.user32
        hwnd = user32.GetParent(self.win.winfo_id()) or self.win.winfo_id()
        GWL_EXSTYLE, WS_EX_LAYERED, WS_EX_TRANSPARENT = -20, 0x80000, 0x20
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT)
        WDA_EXCLUDEFROMCAPTURE = 0x11              # Windows 10 (2004) and newer
        return bool(user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))

    def square_center(self, square, white_at_bottom):
        row, col = square_to_screen(square, white_at_bottom)
        sw, sh = self.region[2] / 8, self.region[3] / 8
        return (col + 0.5) * sw, (row + 0.5) * sh, sw, sh

    def show_move(self, move, white_at_bottom):
        self.clear()
        for sq in (move.from_square, move.to_square):
            cx, cy, sw, sh = self.square_center(sq, white_at_bottom)
            self.canvas.create_rectangle(cx - sw / 2 + 3, cy - sh / 2 + 3, cx + sw / 2 - 3,
                                         cy + sh / 2 - 3, outline=HIGHLIGHT_COLOR,
                                         width=HIGHLIGHT_WIDTH)
        x1, y1, _, _ = self.square_center(move.from_square, white_at_bottom)
        x2, y2, _, _ = self.square_center(move.to_square, white_at_bottom)
        self.canvas.create_line(x1, y1, x2, y2, fill=ARROW_COLOR, width=ARROW_WIDTH,
                                arrow=self.tk.LAST,
                                arrowshape=(ARROW_WIDTH * 2.5, ARROW_WIDTH * 3, ARROW_WIDTH * 1.2))

    def clear(self):
        self.canvas.delete("all")

    def keep_on_top(self):
        self.win.attributes("-topmost", True)
        self.win.lift()


def draw_debug(board_img, grid, white_at_bottom, move=None):
    """Board image with the grid, square names and recognised pieces drawn on.
    Green label = confident, red = below MATCH_THRESHOLD, grey numbers show the
    occupancy stats (outline%/piece%) for empty squares."""
    h, w = board_img.shape[:2]
    scale = max(1.0, 640 / w)
    img = cv2.resize(board_img, None, fx=scale, fy=scale)
    sw, sh = img.shape[1] / 8, img.shape[0] / 8
    for i in range(9):
        cv2.line(img, (int(i * sw), 0), (int(i * sw), img.shape[0]), (255, 0, 255), 1)
        cv2.line(img, (0, int(i * sh)), (img.shape[1], int(i * sh)), (255, 0, 255), 1)

    def text(t, x, y, colour, size=0.45):
        cv2.putText(img, t, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 3)
        cv2.putText(img, t, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, colour, 1)

    for r in range(8):
        for c in range(8):
            cell = grid[r][c]
            x0, y0 = c * sw, r * sh
            text(chess.square_name(screen_to_square(r, c, white_at_bottom)),
                 x0 + 3, y0 + sh - 4, (200, 200, 200), 0.35)
            if cell["label"]:
                ok = cell["score"] >= MATCH_THRESHOLD
                text(cell["label"], x0 + 3, y0 + 14, (0, 255, 0) if ok else (0, 0, 255), 0.5)
                text(f"{cell['score']:.2f}", x0 + sw - 34, y0 + 14,
                     (0, 255, 0) if ok else (0, 0, 255), 0.35)
            else:
                o, p = cell["stats"]
                text(f"{o*100:.0f}/{p*100:.0f}", x0 + 3, y0 + 14, (160, 160, 160), 0.35)
    if move is not None:
        pts = []
        for sq in (move.from_square, move.to_square):
            row, col = square_to_screen(sq, white_at_bottom)
            pts.append((int((col + 0.5) * sw), int((row + 0.5) * sh)))
        cv2.arrowedLine(img, pts[0], pts[1], (255, 180, 0), 6, tipLength=0.25)
    return img


# =============================================================================
# 7. Modes
# =============================================================================

def require_templates():
    templates = load_templates()
    if not templates:
        sys.exit(f"No templates found in '{TEMPLATE_DIR}/'. Create them first with:\n"
                 "  py collect_templates.py image.png --auto-start")
    labels = sorted({t["label"] for t in templates})
    print(f"Loaded {len(templates)} templates covering {len(labels)} piece types.")
    missing = [l for l in PIECE_LABELS if l not in labels]
    if missing:
        print(f"  Warning: no templates for {missing}")
    return templates


def print_board(board, my_color):
    """ASCII board (uppercase = white), shown from your side."""
    if my_color == chess.BLACK:
        board = board.transform(chess.flip_vertical).transform(chess.flip_horizontal)
    print(board)


def run_image(path, turn):
    """Analyse a single saved screenshot."""
    templates = require_templates()
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Could not read {path}")
    region = detect_board(img)
    if region is None:
        print("Board not found automatically - click the corners instead.")
        region = click_region(img)
        if region is None:
            return
    print(f"Board found at x={region[0]} y={region[1]} size={region[2]}x{region[3]}")
    board_img = crop(img, region)
    grid, _ = recognize_board(board_img, templates)
    wab = white_is_at_bottom(grid)
    my_color = chess.WHITE if wab else chess.BLACK
    move = None

    problems = unsure_squares(grid, wab)
    if problems:
        print("Unsure about: " + ", ".join(f"{n} ({c['label']}? {c['score']:.2f})"
                                           for n, c in problems))
    forced = None if turn is None else (chess.WHITE if turn == "white" else chess.BLACK)
    board, status = build_fresh_board(grid_to_piece_map(grid, wab), my_color, forced)
    if board is None:
        print(f"The recognised position is not legal: {status!r}")
    else:
        print(f"You are {'white' if wab else 'black'}.  FEN: {board.fen()}")
        print_board(board, my_color)
        if board.is_game_over():
            print("Game over:", board.result())
        else:
            engine = open_engine()
            try:
                move, score = best_move(engine, board)
                side = "White" if board.turn == chess.WHITE else "Black"
                print(f"\n{side} to move. Best move: {describe_move(board, move)}")
                print(f"Evaluation for {side.lower()}: {describe_score(score)}")
            finally:
                engine.quit()

    dbg = draw_debug(board_img, grid, wab, move)
    cv2.imwrite("debug_output.png", dbg)
    print("Saved debug_output.png. Press any key in the image window to close.")
    cv2.imshow("RoChess debug", dbg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def open_capture():
    """Screen capture object (newer mss versions renamed mss.mss to mss.MSS)."""
    import mss
    return mss.MSS() if hasattr(mss, "MSS") else mss.mss()


def grab_monitor(sct):
    mon = sct.monitors[MONITOR_INDEX]
    shot = cv2.cvtColor(np.array(sct.grab(mon)), cv2.COLOR_BGRA2BGR)
    return shot, mon


def run_calibrate():
    with open_capture() as sct:
        print("Taking a screenshot in 3 seconds - switch to Roblox with the board visible...")
        time.sleep(3)
        shot, mon = grab_monitor(sct)
    hint = detect_board(shot)
    region = click_region(shot, hint=hint)
    if region is None:
        print("Calibration cancelled.")
        return
    if not looks_square(region):
        print(f"That area is {region[2]}x{region[3]} pixels, but the board is square. "
              "Click the board's outer TOP-LEFT and BOTTOM-RIGHT corners and try again.")
        return
    absolute = {"left": region[0] + mon["left"], "top": region[1] + mon["top"],
                "width": region[2], "height": region[3]}
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(absolute, f, indent=2)
    print(f"Saved {CALIBRATION_FILE}: {absolute}")


def looks_square(region):
    """A real board is square; anything far from that is a bad calibration."""
    w, h = region[2], region[3]
    return w >= 80 and h >= 80 and 0.9 < w / h < 1.1


def resolve_region(sct):
    """BOARD_REGION constant > calibration.json > automatic detection."""
    if BOARD_REGION:
        return tuple(BOARD_REGION)
    if os.path.isfile(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            d = json.load(f)
        region = (d["left"], d["top"], d["width"], d["height"])
        if looks_square(region):
            return region
        print(f"Ignoring {CALIBRATION_FILE}: {d['width']}x{d['height']} is not square. "
              "Delete it or re-run --calibrate.")
    # Keep looking until the board is visible (e.g. not hidden behind another window).
    waiting = False
    while True:
        shot, mon = grab_monitor(sct)
        found = detect_board(shot)
        if found is not None:
            return (found[0] + mon["left"], found[1] + mon["top"], found[2], found[3])
        if not waiting:
            print("Board not visible yet - bring the Roblox window to the front "
                  "(Ctrl+C to quit, or use --calibrate)...")
            waiting = True
        time.sleep(1)


def run_live(debug):
    import tkinter as tk

    templates = require_templates()
    sct = open_capture()
    region = resolve_region(sct)
    print(f"Watching board at x={region[0]} y={region[1]} size={region[2]}x{region[3]}")
    engine = open_engine()
    tracker = GameTracker()

    root = tk.Tk()
    root.withdraw()                         # the root window itself stays hidden
    overlay = Overlay(root, region) if SHOW_OVERLAY else None
    if overlay and not overlay.hidden_from_capture:
        print("Note: this Windows version can't hide the overlay from capture; "
              "it will be hidden briefly during each screenshot.")

    state = {"candidate": None, "count": 0, "acted": None, "move": None, "wab": True}
    cap = {"left": region[0], "top": region[1], "width": region[2], "height": region[3]}

    def capture():
        if overlay and not overlay.hidden_from_capture:
            overlay.win.withdraw()
            root.update()
        img = cv2.cvtColor(np.array(sct.grab(cap)), cv2.COLOR_BGRA2BGR)
        if overlay and not overlay.hidden_from_capture:
            overlay.win.deiconify()
        return img

    def on_new_position(grid):
        wab = white_is_at_bottom(grid)
        state["wab"], state["move"] = wab, None
        my_color = chess.WHITE if wab else chess.BLACK
        if overlay:
            overlay.clear()
        print("\n" + "=" * 60)
        problems = unsure_squares(grid, wab)
        if problems:
            print("Unsure about: " + ", ".join(f"{n} ({c['label']}? {c['score']:.2f})"
                                               for n, c in problems))
            print("Skipping this frame. Save it with S in the debug window and add "
                  "templates with collect_templates.py")
            return
        board, how, problem = tracker.update(grid_to_piece_map(grid, wab), my_color)
        if board is None:
            print(f"Recognised position is not legal ({problem!r}) - check the debug view.")
            return
        print(f"[{how}]  FEN: {board.fen()}")
        if debug:
            print_board(board, my_color)
        if board.is_game_over():
            print("Game over:", board.result())
            return
        if board.turn != my_color:
            print("Waiting for the opponent to move...")
            return
        move, score = best_move(engine, board)
        state["move"] = move
        print(f"YOUR MOVE: {describe_move(board, move)}")
        print(f"Evaluation: {describe_score(score)}")
        if overlay:
            overlay.show_move(move, wab)

    def tick():
        board_img = capture()
        grid, _ = recognize_board(board_img, templates)
        key = grid_key(grid)
        if key == state["candidate"]:
            state["count"] += 1
        else:
            state["candidate"], state["count"] = key, 1
        if state["count"] >= STABLE_FRAMES and key != state["acted"]:
            state["acted"] = key
            on_new_position(grid)
        if overlay:
            overlay.keep_on_top()
        if debug:
            cv2.imshow("RoChess debug", draw_debug(board_img, grid, state["wab"], state["move"]))
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                root.quit()
                return
            if k == ord("s"):
                os.makedirs("screenshots", exist_ok=True)
                path = os.path.join("screenshots", time.strftime("board_%Y%m%d_%H%M%S.png"))
                cv2.imwrite(path, board_img)
                print(f"Saved {path}")
        root.after(LOOP_DELAY_MS, tick)

    def on_error(exc, val, tb):
        # Ctrl+C arrives inside a Tk callback; turn it into a clean exit.
        if issubclass(exc, KeyboardInterrupt):
            root.quit()
        else:
            import traceback
            traceback.print_exception(exc, val, tb)

    root.report_callback_exception = on_error
    if debug:
        cv2.namedWindow("RoChess debug")
        cv2.moveWindow("RoChess debug", region[0] + region[2] + 20, max(0, region[1]))
    print("Running. Press Ctrl+C here (or Q in the debug window) to stop.")
    root.after(0, tick)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        engine.quit()
        cv2.destroyAllWindows()
        print("Stopped.")


def main():
    ap = argparse.ArgumentParser(description="RoChess board reader + Stockfish helper")
    ap.add_argument("--image", help="analyse a saved screenshot instead of the live screen")
    ap.add_argument("--turn", choices=["white", "black"],
                    help="with --image: whose turn it is (default: yours)")
    ap.add_argument("--calibrate", action="store_true", help="mark the board on screen once")
    ap.add_argument("--debug", action="store_true", help="show the recognition debug window")
    args = ap.parse_args()

    make_dpi_aware()
    if args.calibrate:
        run_calibrate()
    elif args.image:
        run_image(args.image, args.turn)
    else:
        run_live(args.debug or DEBUG)


if __name__ == "__main__":
    main()
