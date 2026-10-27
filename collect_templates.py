"""
collect_templates.py - turn your own screenshots into piece templates.

Each template is one square cut out of a screenshot, saved as
    templates/<label>_<shade>_<source>_<square>.png
where <label> is colour + piece (wP wN wB wR wQ wK bP bN bB bR bQ bK) and
<shade> is the square colour it was sitting on (light / dark).

Usage:
    # 1. Starting position: labels all 32 pieces automatically
    py collect_templates.py image.png --auto-start

    # 2. Any other screenshot: label squares by typing, e.g. "wq" or "bn"
    py collect_templates.py screenshots/board_20261025_120000.png

    # Only dump the 64 squares as separate PNGs so you can look at them
    py collect_templates.py image.png --dump

Options:
    --play-as white|black   which side is at the bottom (default: auto from the
                            white king, or white with --auto-start)
    --all                   in labelling mode, ask about all 64 squares, not
                            just the ones that look occupied
"""

import argparse
import os

import chess
import cv2

import rochess_assistant as ra


def load_board(path):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Could not read {path}")
    region = ra.detect_board(img)
    if region is None:
        print("Board not found automatically - click the corners.")
        region = ra.click_region(img)
        if region is None:
            raise SystemExit("Cancelled.")
    print(f"Board at x={region[0]} y={region[1]} size={region[2]}x{region[3]}")
    board_img = ra.crop(img, region)
    squares = ra.split_squares(board_img)
    return board_img, squares, ra.board_palette(squares)


def parse_label(text):
    """'wn', 'WN', 'w n' -> 'wN'. Returns None if not a valid label."""
    t = text.replace(" ", "").lower()
    if len(t) == 2 and t[0] in "wb" and t[1] in "pnbrqk":
        return t[0] + t[1].upper()
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--auto-start", action="store_true",
                    help="the image shows the starting position; label automatically")
    ap.add_argument("--play-as", choices=["white", "black"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dump", action="store_true", help="just save all 64 squares to squares_dump/")
    args = ap.parse_args()

    ra.make_dpi_aware()
    board_img, squares, palette = load_board(args.image)
    stem = os.path.splitext(os.path.basename(args.image))[0].replace("_", "-")

    # Which way round is the board?
    if args.play_as:
        wab = args.play_as == "white"
    elif args.auto_start:
        wab = True
    else:
        templates = ra.load_templates()
        grid, _ = ra.recognize_board(board_img, templates) if templates else (None, None)
        wab = ra.white_is_at_bottom(grid) if grid else True
    print(f"Assuming {'white' if wab else 'black'} is at the bottom of the screen.")

    if args.dump:
        os.makedirs("squares_dump", exist_ok=True)
        for r in range(8):
            for c in range(8):
                name = chess.square_name(ra.screen_to_square(r, c, wab))
                cv2.imwrite(os.path.join("squares_dump", f"{stem}_{name}.png"), squares[r][c])
        print("Saved 64 squares to squares_dump/")
        return

    if args.auto_start:
        start = chess.Board()
        saved = 0
        for r in range(8):
            for c in range(8):
                sq = ra.screen_to_square(r, c, wab)
                piece = start.piece_at(sq)
                if piece is None:
                    continue
                label = ra.piece_to_label(piece)
                ra.save_template(squares[r][c], label, ra.square_shade(r, c), palette,
                                 f"{stem}_{chess.square_name(sq)}")
                saved += 1
        print(f"Saved {saved} templates to {ra.TEMPLATE_DIR}/")
        print("Note: the starting position never shows the kings and queens on both "
              "square colours. That's usually fine, but if one is misread later, "
              "label it from a mid-game screenshot.")
        return

    # Interactive labelling. The current best guess (if any templates exist) is
    # shown so you only need to type when the guess is wrong.
    templates = ra.load_templates()
    print("\nFor each square: type a label (wp wn wb wr wq wk bp bn bb br bq bk) to save "
          "it as a template,\n  press Enter to skip, or type 'quit' to stop.\n")
    saved = 0
    for r in range(8):
        for c in range(8):
            sq_img = ra.normalise_square(squares[r][c])
            stats = ra.occupancy_stats(sq_img, palette)
            if not args.all and not ra.is_occupied(stats):
                continue
            shade = ra.square_shade(r, c)
            name = chess.square_name(ra.screen_to_square(r, c, wab))
            guess = ""
            if templates:
                label, score = ra.classify(ra.square_feature(sq_img, palette), shade, templates)
                guess = f"  guess: {label} ({score:.2f})"
            cv2.imshow("square", cv2.resize(squares[r][c], (240, 240), interpolation=cv2.INTER_NEAREST))
            cv2.waitKey(100)
            while True:
                answer = input(f"{name} [{shade}]{guess} > ").strip()
                if answer == "" or answer.lower() == "quit":
                    break
                label = parse_label(answer)
                if label:
                    path = ra.save_template(squares[r][c], label, shade, palette, f"{stem}_{name}")
                    print(f"   saved {path}")
                    saved += 1
                    break
                print("   not a label - use colour + piece, e.g. wq or bn")
            if answer.lower() == "quit":
                cv2.destroyAllWindows()
                print(f"Saved {saved} templates.")
                return
    cv2.destroyAllWindows()
    print(f"Saved {saved} templates.")


if __name__ == "__main__":
    main()
