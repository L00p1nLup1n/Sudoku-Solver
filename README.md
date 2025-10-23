# Sudoku-Solver

Sudoku-Solver is a small Python project that extracts digits from Sudoku puzzle images and solves the puzzle.

This project aims to demostrate the usage of OpenCV and Tesseract in Python.

## Key functionality:

- A Tkinter GUI for loading or pasting Sudoku images and displaying/solving boards.
- An OCR pipeline that extracts individual cell images and recognizes digits (prefers OpenCV-based preprocessing; falls back to a PIL-based split).
- A backtracking Sudoku solver that fills a 9x9 board and validates puzzles.
- Includes [tessdata](https://github.com/tesseract-ocr/tessdata) so OCR can be run with packaged language models when available.

See the `python/sudoku` package for implementation details of each component.
- Solve: uses the backtracking solver in `python/sudoku/solver.py`.
- Reset: clear the board.
- Paste Image: Load a Sudoku board from the clipboard image
- Load Image: Load a Sudoku board from the directory (preferably `upload/`)