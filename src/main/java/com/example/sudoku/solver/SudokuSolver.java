package com.example.sudoku.solver;

public class SudokuSolver {

    private static final int SIZE = 9;

    public boolean solve(int[][] board) {
        for (int row = 0; row < SIZE; row++) {
            for (int col = 0; col < SIZE; col++) {
                if (board[row][col] == 0) {
                    for (int num = 1; num <= SIZE; num++) {
                        if (isValid(board, row, col, num)) {
                            board[row][col] = num;
                            if (solve(board)) return true;
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
            if (board[row][c] == num) return false;
        }
        for (int r = 0; r < SIZE; r++) {
            if (board[r][col] == num) return false;
        }
        int boxRowStart = row - row % 3;
        int boxColStart = col - col % 3;
        for (int r = boxRowStart; r < boxRowStart + 3; r++) {
            for (int c = boxColStart; c < boxColStart + 3; c++) {
                if (board[r][c] == num) return false;
            }
        }
        return true;
    }
}