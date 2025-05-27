package com.example.sudoku;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;

public class SudokuSolverGUI extends JFrame {

    private static final int SIZE = 9;
    private final JTextField[][] cells = new JTextField[SIZE][SIZE];

    public SudokuSolverGUI() {
        setTitle("Sudoku Solver");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(600, 600);
        setLayout(new BorderLayout());

        JPanel gridPanel = new JPanel(new GridLayout(SIZE, SIZE));
        Font font = new Font("SansSerif", Font.BOLD, 20);

        // Create 9x9 grid of text fields
        for (int row = 0; row < SIZE; row++) {
            for (int col = 0; col < SIZE; col++) {
                JTextField cell = new JTextField();
                cell.setHorizontalAlignment(JTextField.CENTER);
                cell.setFont(font);
                cell.setDocument(new JTextFieldLimit(1)); // limit input length to 1
                // Set border for 3x3 blocks to look nicer
                int top = (row % 3 == 0) ? 4 : 1;
                int left = (col % 3 == 0) ? 4 : 1;
                int bottom = ((row + 1) % 3 == 0) ? 4 : 1;
                int right = ((col + 1) % 3 == 0) ? 4 : 1;
                cell.setBorder(BorderFactory.createMatteBorder(top, left, bottom, right, Color.BLACK));

                // Allow only digits 1-9 or empty
                cell.addKeyListener(new KeyAdapter() {
                    @Override
                    public void keyTyped(KeyEvent e) {
                        char c = e.getKeyChar();
                        if (!(c >= '1' && c <= '9') && c != '\b') {
                            e.consume();
                        }
                    }
                });

                cells[row][col] = cell;
                gridPanel.add(cell);
            }
        }

        // Buttons panel
        JPanel buttonsPanel = new JPanel();

        JButton solveButton = new JButton("Solve");
        solveButton.addActionListener(e -> solveSudoku());

        JButton resetButton = new JButton("Reset");
        resetButton.addActionListener(e -> resetBoard());

        buttonsPanel.add(solveButton);
        buttonsPanel.add(resetButton);

        add(gridPanel, BorderLayout.CENTER);
        add(buttonsPanel, BorderLayout.SOUTH);

        setLocationRelativeTo(null);
        setVisible(true);
    }

    private void resetBoard() {
        for (int row = 0; row < SIZE; row++) {
            for (int col = 0; col < SIZE; col++) {
                cells[row][col].setText("");
            }
        }
    }

    private void solveSudoku() {
        int[][] board = new int[SIZE][SIZE];

        // Read input from text fields
        for (int row = 0; row < SIZE; row++) {
            for (int col = 0; col < SIZE; col++) {
                String text = cells[row][col].getText();
                if (text.isEmpty()) {
                    board[row][col] = 0;
                } else {
                    try {
                        int val = Integer.parseInt(text);
                        if (val < 1 || val > 9)
                            throw new NumberFormatException();
                        board[row][col] = val;
                    } catch (NumberFormatException ex) {
                        JOptionPane.showMessageDialog(this,
                                "Invalid input at row " + (row + 1) + ", column " + (col + 1),
                                "Input Error", JOptionPane.ERROR_MESSAGE);
                        return;
                    }
                }
            }
        }

        if (solve(board)) {
            // Update GUI with solution
            for (int row = 0; row < SIZE; row++) {
                for (int col = 0; col < SIZE; col++) {
                    cells[row][col].setText(Integer.toString(board[row][col]));
                }
            }
        } else {
            JOptionPane.showMessageDialog(this,
                    "No solution exists for the given puzzle.",
                    "No Solution", JOptionPane.WARNING_MESSAGE);
        }
    }

    // Backtracking Sudoku solver
    private boolean solve(int[][] board) {
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

    private boolean isValid(int[][] board, int row, int col, int num) {
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

    // Limit input length for text fields
    private static class JTextFieldLimit extends javax.swing.text.PlainDocument {
        private final int limit;

        JTextFieldLimit(int limit) {
            super();
            this.limit = limit;
        }

        @Override
        public void insertString(int offset, String str, javax.swing.text.AttributeSet attr)
                throws javax.swing.text.BadLocationException {
            if (str == null)
                return;
            if ((getLength() + str.length()) <= limit) {
                super.insertString(offset, str, attr);
            }
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(SudokuSolverGUI::new);
    }
}
