package com.example.sudoku.solver;

public class SudokuSolver {

    public interface StepSolverCallback {
        void onStep(int row, int col, int num);
    }

    private static final int SIZE = 9;

    public boolean solve(int[][] board) {
        for (int row = 0; row < SIZE; row++) {
            for (int col = 0; col < SIZE; col++) {
                if (board[row][col] == 0) {
                    for (int num = 1; num <= SIZE; num++) {
                        if (isValid(board, row, col, num)) {
                            board[row][col] = num;
                            if (solve(board))
                                return true;
                            board[row][col] = 0;
                        }
                    }
                    return false;
                }
            }
        }
        return true;
    }

    public boolean isValid(int[][] board, int row, int col, int num) {
        for (int c = 0; c < SIZE; c++) {
            if (board[row][c] == num)
                return false;
        }
        for (int r = 0; r < SIZE; r++) {
            if (board[r][col] == num)
                return false;
        }
        int boxRowStart = row - row % 3;
        int boxColStart = col - col % 3;
        for (int r = boxRowStart; r < boxRowStart + 3; r++) {
            for (int c = boxColStart; c < boxColStart + 3; c++) {
                if (board[r][c] == num)
                    return false;
            }
        }
        return true;
    }

    public boolean isBoardValid(int[][] board) {
        int size = board.length;
        int clueCount = 0; // To count the number of clues (non-empty cells)

        // Check rows and columns for conflicts
        for (int i = 0; i < size; i++) {
            boolean[] rowCheck = new boolean[size + 1];
            boolean[] colCheck = new boolean[size + 1];
            for (int j = 0; j < size; j++) {
                int rowVal = board[i][j];
                int colVal = board[j][i];

                // Check if there's a conflict in the row
                if (rowVal != 0) {
                    if (rowCheck[rowVal])
                        return false;
                    rowCheck[rowVal] = true;
                    clueCount++; // Count clue
                }

                // Check if there's a conflict in the column
                if (colVal != 0) {
                    if (colCheck[colVal])
                        return false;
                    colCheck[colVal] = true;
                    clueCount++; // Count clue
                }
            }
        }

        // Check 3x3 boxes for conflicts
        for (int boxRow = 0; boxRow < 3; boxRow++) {
            for (int boxCol = 0; boxCol < 3; boxCol++) {
                boolean[] boxCheck = new boolean[size + 1];
                for (int r = boxRow * 3; r < boxRow * 3 + 3; r++) {
                    for (int c = boxCol * 3; c < boxCol * 3 + 3; c++) {
                        int val = board[r][c];
                        if (val != 0) {
                            if (boxCheck[val])
                                return false;
                            boxCheck[val] = true;
                            clueCount++; // Count clue
                        }
                    }
                }
            }
        }

        // Check if there are at least 17 clues
        return clueCount >= 17;
    }

    public boolean solveStepwise(int[][] board, StepSolverCallback callback) {
        for (int row = 0; row < SIZE; row++) {
            for (int col = 0; col < SIZE; col++) {
                if (board[row][col] == 0) {
                    for (int num = 1; num <= SIZE; num++) {
                        if (isValid(board, row, col, num)) {
                            board[row][col] = num;
                            callback.onStep(row, col, num);

                            if (solveStepwise(board, callback))
                                return true;

                            board[row][col] = 0;
                            callback.onStep(row, col, 0); // backtrack step
                        }
                    }
                    return false;
                }
            }
        }
        return true;
    }
}