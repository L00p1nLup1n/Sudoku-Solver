
# Sudoku Solver GUI

A Java application that solves Sudoku puzzles by manual input or by loading an image of a Sudoku board. It uses OpenCV for image processing and Tesseract OCR (via Tess4J) for digit recognition.

---

## Features

- Enter Sudoku numbers manually or load a photo to automatically extract digits.
- Solve puzzles using a backtracking algorithm.
- Reset the board to clear all inputs.
- Improved image preprocessing for better OCR accuracy.
- Cross-platform support with included language data as a Git submodule.

---

## Setup Instructions

### 1. Clone the repository with submodules

```bash
git clone --recurse -submodules https://github.com/L00p1nLup1n/Sudoku-Solver.git
````

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

This ensures the OCR language data (`tessdata`) is included.

---

### 2. Build the project

Requires Java 17+ and Maven 3.9.9+.

```bash
mvn clean package
```

---

### 3. Run the application

```bash
java -jar target/sudoku-solver-1.0-SNAPSHOT.jar
```

---

## Usage

* **Manual Input:** Type digits (1-9) into the grid.
* **Load Image:** Import a Sudoku photo or paste one from clipboard to auto-fill the grid.
* **Solve:** Click to solve the puzzle.
* **Reset:** Clear the grid.

---

## Notes

* The `upload` folder contains a placeholder file (`.gitkeep`) to ensure it is tracked by Git.
* Ensure the `tessdata` folder is fully downloaded as a Git submodule and the path is correctly set.
* To update submodules after pulling changes, run:

```bash
git submodule update --init --recursive
```

---

## Troubleshooting

* **OCR crashes or errors:** Confirm native libraries and Tesseract data are correctly installed and configured.
* **Large files:** Build artifacts such as JAR files are ignored via `.gitignore` and should not be committed to the repo.

---

## License

Open source and free to use.

---

## Acknowledgments

* OpenCV for image processing.
* Tess4J for Tesseract OCR integration.

---

Enjoy solving Sudoku puzzles!


