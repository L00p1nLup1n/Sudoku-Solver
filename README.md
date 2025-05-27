````markdown
# Sudoku Solver GUI

A simple Java Swing application that allows users to manually input Sudoku puzzles, solve them using a backtracking algorithm, and reset the board.

## Features

- **Manual Input:** Enter digits (1–9) into a 9x9 grid.
- **Solve Puzzle:** Automatically solves the puzzle if a solution exists.
- **Reset Board:** Clear the entire board with a single click.
- **Input Validation:** Only digits 1 to 9 allowed, with input length limited to one character per cell.
- **User-friendly UI:** Grid with bold borders every 3 cells to clearly show Sudoku blocks.

## Requirements

- Java 17 or higher
- Maven 3.9.9 or higher

## How to Build and Run

1. Clone the repository or download the source code.

2. Build the project using Maven:

   ```bash
   mvn clean package
````

3. Run the generated JAR:

   ```bash
   java -jar target/sudoku-solver-1.0-SNAPSHOT.jar
   ```

## Usage

* Enter digits (1 to 9) in the Sudoku grid cells.
* Click **Solve** to solve the puzzle.
* Click **Reset** to clear the board and start over.
* If invalid input is detected, an error message will appear.
* If no solution exists, a warning will be shown.

## Project Structure

* `src/main/java/com/example/sudoku/SudokuSolverGUI.java`: Main GUI and solver logic.
* `pom.xml`: Maven configuration file.

## License

This project is open source and free to use.

---

Enjoy solving Sudoku puzzles effortlessly!

```


