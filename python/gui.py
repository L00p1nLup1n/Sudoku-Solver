import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2


class SudokuGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sudoku Solver (Python)")
        self.cells = [[tk.StringVar() for _ in range(9)] for _ in range(9)]
        self._last_image = None
        self._build_ui()

    def _build_ui(self):
        frame = tk.Frame(self)
        frame.pack(padx=8, pady=8)
        vcmd = self.register(self._validate_cells)

        for br in range(3):
            for bc in range(3):
                block = tk.Frame(frame, bd=3, relief="solid")
                block.grid(
                    row=br,
                    column=bc,
                    padx=(0 if bc == 0 else 4),
                    pady=(0 if br == 0 else 4),
                )
                for r in range(3):
                    for c in range(3):
                        row_idx = br * 3 + r
                        col_idx = bc * 3 + c
                        e = tk.Entry(
                            block,
                            width=2,
                            font=(None, 18),
                            textvariable=self.cells[row_idx][col_idx],
                            justify="center",
                            bd=1,
                            relief="solid",
                            validate="key",
                            validatecommand=(vcmd, "%P"),
                        )
                        # small internal padding to make cells easier to click
                        e.grid(row=r, column=c, ipadx=6, ipady=6, padx=0, pady=0)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=6)
        tk.Button(
            btn_frame, text="Paste Image", command=self.paste_image_from_clipboard
        ).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Load Image", command=self.load_image).pack(
            side="left", padx=6
        )
        tk.Button(btn_frame, text="Solve", command=self.solve_board).pack(
            side="left", padx=6
        )
        tk.Button(btn_frame, text="Reset", command=self.reset_board).pack(
            side="left", padx=6
        )

    def get_board(self):
        board = [[0] * 9 for _ in range(9)]
        for r in range(9):
            for c in range(9):
                v = self.cells[r][c].get().strip()
                board[r][c] = int(v) if v.isdigit() else 0
        return board

    def set_board(self, board):
        for r in range(9):
            for c in range(9):
                self.cells[r][c].set(str(board[r][c]) if board[r][c] != 0 else "")

    def reset_board(self):
        self.set_board([[0] * 9 for _ in range(9)])

    def _validate_cells(self, proposed):
        if proposed == "":
            return True
        if len(proposed) == 1 and proposed.isdigit and 1 <= int(proposed):
            return True
        return False

    def solve_board(self):
        board = self.get_board()

        from solver import is_board_valid, solve

        if not is_board_valid(board):
            messagebox.showerror(
                "Invalid", "Board validation failed (conflicts or too few clues)"
            )
            return
        if solve(board):
            self.set_board(board)
        else:
            messagebox.showwarning("No solution", "No solution found for this board")

    def load_image(self):
        # start in repo's upload/ folder if it exists to make selection quicker
        start_dir = (
            os.path.join(os.getcwd(), "upload")
            if os.path.isdir(os.path.join(os.getcwd(), "upload"))
            else None
        )
        path = filedialog.askopenfilename(
            title="Open Sudoku image",
            initialdir=start_dir,
            filetypes=[("Images", "*.png;*.jpg;*.jpeg")],
        )
        if not path:
            return

        from PIL import Image

        img = Image.open(path).convert("L")
        self._last_image = img
        try:
            self.run_ocr_fill()
        except Exception as e:
            # still show loaded message if OCR fails
            messagebox.showinfo(
                "Loaded", f"Image loaded: {os.path.basename(path)}\nOCR failed: {e}"
            )

    def run_ocr_fill(self):
        if self._last_image is None:
            messagebox.showwarning("No image", "Load an image first")
            return

        board = [[0] * 9 for _ in range(9)]
        import numpy as np

        img_bgr = cv2.cvtColor(
            np.array(self._last_image.convert("RGB")), cv2.COLOR_RGB2BGR
        )
        from cell_splitter import split_cells_from_array as split_cells
        from image_preprocessor import preprocess_sudoku_image_from_array as preprocess

        warped = preprocess(img_bgr)
        cells = split_cells(warped)

        from ocr import ocr_image_to_digit as ocr_fn

        for i, cv_cell in enumerate(cells):
            # cv_cell is a 28x28 grayscale numpy array
            board[i // 9][i % 9] = ocr_fn(cv_cell)

        self.set_board(board)

    def paste_image_from_clipboard(self):
        try:
            from PIL import ImageGrab
        except Exception:
            ImageGrab = None

        if ImageGrab is None:
            messagebox.showerror(
                "Clipboard error",
                "Pillow ImageGrab not available. Install pillow and ensure clipboard access is allowed.",
            )
            return

        try:
            img = ImageGrab.grabclipboard()
        except Exception as e:
            messagebox.showerror(
                "Clipboard error", f"Failed to read clipboard image: {e}"
            )
            return

        if img is None:
            messagebox.showinfo("No image", "No image found in clipboard")
            return

        self._last_image = img
        try:
            self.run_ocr_fill()
        except Exception as e:
            messagebox.showerror("OCR error", f"Error running OCR: {e}")


def run_gui():
    app = SudokuGUI()
    app.mainloop()


if __name__ == "__main__":
    # allow running the module directly: ensure package import works
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    run_gui()
