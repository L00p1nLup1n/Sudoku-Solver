package com.example.sudoku.app;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.Toolkit;
import java.awt.datatransfer.DataFlavor;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import com.example.sudoku.image.CellSplitter;
import com.example.sudoku.image.ImagePreprocessor;
import com.example.sudoku.ocr.ImageConverter;
import com.example.sudoku.ocr.OCRProcessor;
import com.example.sudoku.solver.SudokuSolver;
import com.example.sudoku.util.JTextFieldLimit;

import net.sourceforge.tess4j.ITesseract;
import net.sourceforge.tess4j.Tesseract;
import nu.pattern.OpenCV;

public class SudokuSolverGUI extends JFrame {

    private static final int SIZE = 9;
    private final JTextField[][] cells = new JTextField[SIZE][SIZE];
    private File lastDirectory = null;

    private final SudokuSolver solver = new SudokuSolver();
    private final OCRProcessor ocrProcessor = new OCRProcessor();
    private final ImagePreprocessor imagePreprocessor = new ImagePreprocessor();
    private final CellSplitter cellSplitter = new CellSplitter();

    public SudokuSolverGUI() {
        setTitle("Sudoku Solver");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(600, 600);
        setLayout(new BorderLayout());

        JPanel gridPanel = new JPanel(new GridLayout(SIZE, SIZE));
        Font font = new Font("SansSerif", Font.BOLD, 20);

        for (int row = 0; row < SIZE; row++) {
            for (int col = 0; col < SIZE; col++) {
                JTextField cell = new JTextField();
                cell.setHorizontalAlignment(JTextField.CENTER);
                cell.setFont(font);
                cell.setDocument(new JTextFieldLimit(1));

                int top = (row % 3 == 0) ? 4 : 1;
                int left = (col % 3 == 0) ? 4 : 1;
                int bottom = ((row + 1) % 3 == 0) ? 4 : 1;
                int right = ((col + 1) % 3 == 0) ? 4 : 1;
                cell.setBorder(BorderFactory.createMatteBorder(top, left, bottom, right, Color.BLACK));

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

        JPanel buttonsPanel = new JPanel();

        JButton pasteImageButton = new JButton("Paste Sudoku Image");
        pasteImageButton.addActionListener(e -> pasteSudokuImage());

        JButton loadImageButton = new JButton("Load Sudoku Image");
        loadImageButton.addActionListener(e -> loadSudokuImage());

        JButton solveButton = new JButton("Solve");
        solveButton.addActionListener(e -> solveSudoku());

        JButton resetButton = new JButton("Reset");
        resetButton.addActionListener(e -> resetBoard());

        buttonsPanel.add(pasteImageButton);
        buttonsPanel.add(loadImageButton);
        buttonsPanel.add(solveButton);
        buttonsPanel.add(resetButton);

        add(gridPanel, BorderLayout.CENTER);
        add(buttonsPanel, BorderLayout.SOUTH);

        setLocationRelativeTo(null);
        setVisible(true);
    }

    private void pasteSudokuImage() {
        try {
            var clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
            if (!clipboard.isDataFlavorAvailable(DataFlavor.imageFlavor)) {
                JOptionPane.showMessageDialog(this, "Clipboard does not contain an image", "Error",
                        JOptionPane.ERROR_MESSAGE);
                return;
            }
            BufferedImage bi = (BufferedImage) clipboard.getData(DataFlavor.imageFlavor);

            // Use ImageConverter utility method here:
            Mat src = ImageConverter.bufferedImageToMat(bi);

            Mat warpedGrid = imagePreprocessor.preprocessSudokuImage(src);
            List<Mat> cellImages = cellSplitter.splitCells(warpedGrid);

            ITesseract tesseract = new Tesseract();
            tesseract.setDatapath("tessdata");
            tesseract.setVariable("tessedit_char_whitelist", "123456789");

            int[][] board = ocrProcessor.recognizeDigits(cellImages, tesseract);

            updateGrid(board);

            JOptionPane.showMessageDialog(this, "Pasted image processed. Verify before solving.");

        } catch (Exception e) {
            JOptionPane.showMessageDialog(this, "Failed to process image: " + e.getMessage(), "Error",
                    JOptionPane.ERROR_MESSAGE);
            e.printStackTrace();
        }
    }

    private void loadSudokuImage() {
        String projectRoot = System.getProperty("user.dir");
        File uploadDir = new File(projectRoot, "upload");

        // Create upload folder if missing
        if (!uploadDir.exists()) {
            uploadDir.mkdirs();
        }

        File startDir = (lastDirectory != null && lastDirectory.exists()) ? lastDirectory : uploadDir;

        JFileChooser chooser = new JFileChooser(startDir);
        int result = chooser.showOpenDialog(this);
        if (result != JFileChooser.APPROVE_OPTION)
            return;

        File imageFile = chooser.getSelectedFile();
        lastDirectory = imageFile.getParentFile();

        try {
            Mat src = Imgcodecs.imread(imageFile.getAbsolutePath());
            Mat warped = imagePreprocessor.preprocessSudokuImage(src);
            List<Mat> cellsImages = cellSplitter.splitCells(warped);
            ITesseract tesseract = new Tesseract();
            tesseract.setDatapath("tessdata");
            int[][] board = ocrProcessor.recognizeDigits(cellsImages, tesseract);
            updateGrid(board);
            JOptionPane.showMessageDialog(this, "Sudoku board loaded from image! Verify and edit if needed.");
        } catch (Exception e) {
            JOptionPane.showMessageDialog(this, "Failed to process image: " + e.getMessage(), "Error",
                    JOptionPane.ERROR_MESSAGE);
            e.printStackTrace();
        }
    }

    private void updateGrid(int[][] board) {
        for (int r = 0; r < SIZE; r++) {
            for (int c = 0; c < SIZE; c++) {
                cells[r][c].setText(board[r][c] == 0 ? "" : String.valueOf(board[r][c]));
            }
        }
    }

    private void solveSudoku() {
        int[][] board = new int[SIZE][SIZE];
        for (int r = 0; r < SIZE; r++) {
            for (int c = 0; c < SIZE; c++) {
                String text = cells[r][c].getText();
                board[r][c] = text.isEmpty() ? 0 : Integer.parseInt(text);
            }
        }
        if (!solver.isBoardValid(board)) {
            JOptionPane.showMessageDialog(this, "Invalid board: conflicts detected in input.", "Input Error",
                    JOptionPane.ERROR_MESSAGE);
            return;
        }
        if (solver.solve(board)) {
            updateGrid(board);
        } else {
            JOptionPane.showMessageDialog(this, "No solution exists for the given puzzle.", "Warning",
                    JOptionPane.WARNING_MESSAGE);
        }
    }

    private void resetBoard() {
        for (int r = 0; r < SIZE; r++) {
            for (int c = 0; c < SIZE; c++) {
                cells[r][c].setText("");
            }
        }
    }

    public static void main(String[] args) {
        OpenCV.loadLocally();
        SwingUtilities.invokeLater(SudokuSolverGUI::new);
    }
}
