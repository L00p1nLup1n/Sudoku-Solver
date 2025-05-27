package com.example.sudoku;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
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
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import net.sourceforge.tess4j.ITessAPI;
import net.sourceforge.tess4j.ITesseract;
import net.sourceforge.tess4j.Tesseract;
import nu.pattern.OpenCV;

public class SudokuSolverGUI extends JFrame {

    private static final int SIZE = 9;
    private final JTextField[][] cells = new JTextField[SIZE][SIZE];
    private File lastDirectory = null; // remembers last used folder

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

        JButton loadImageButton = new JButton("Load Sudoku Image");
        loadImageButton.addActionListener(e -> loadSudokuImage());

        buttonsPanel.add(loadImageButton);
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

    private void loadSudokuImage() {
        String projectRoot = System.getProperty("user.dir");
        File uploadDir = new File(projectRoot, "upload");

        // Create 'upload' folder if missing
        if (!uploadDir.exists()) {
            uploadDir.mkdirs();
        }

        // Choose start directory: last used or uploadDir
        File startDir = (lastDirectory != null && lastDirectory.exists()) ? lastDirectory : uploadDir;

        JFileChooser chooser = new JFileChooser(startDir);
        int result = chooser.showOpenDialog(this);
        if (result != JFileChooser.APPROVE_OPTION)
            return;

        File imageFile = chooser.getSelectedFile();
        lastDirectory = imageFile.getParentFile(); // remember folder for next time

        try {
            // 1) Preprocess the image to find and warp the Sudoku grid
            Mat warpedGrid = preprocessSudokuImage(imageFile.getAbsolutePath());

            // 2) Split into 81 individual cell images
            List<Mat> cellImages = splitCells(warpedGrid);

            // 3) Set up Tesseract OCR
            ITesseract tesseract = new Tesseract();
            tesseract.setDatapath("tessdata"); // Path to tessdata folder
            tesseract.setVariable("tessedit_char_whitelist", "123456789");

            // 4) OCR each cell to get digits
            int[][] board = recognizeDigits(cellImages, tesseract);

            // 5) Fill the GUI grid with recognized digits
            for (int r = 0; r < SIZE; r++) {
                for (int c = 0; c < SIZE; c++) {
                    cells[r][c].setText(board[r][c] == 0 ? "" : String.valueOf(board[r][c]));
                }
            }
            JOptionPane.showMessageDialog(this, "Sudoku board loaded from image! Verify and edit if needed.");

        } catch (Exception e) {
            JOptionPane.showMessageDialog(this, "Failed to process image: " + e.getMessage(), "Error",
                    JOptionPane.ERROR_MESSAGE);
            e.printStackTrace();
        }
    }

    private Point[] orderPoints(Point[] pts) {
        Point[] ordered = new Point[4];

        // Sum of x and y will give top-left (smallest sum) and bottom-right (largest
        // sum)
        Arrays.sort(pts, Comparator.comparingDouble(p -> p.x + p.y));
        ordered[0] = pts[0]; // top-left
        ordered[2] = pts[3]; // bottom-right

        // Difference of y - x will give top-right (smallest diff) and bottom-left
        // (largest diff)
        Arrays.sort(pts, Comparator.comparingDouble(p -> p.y - p.x));
        ordered[1] = pts[0]; // top-right
        ordered[3] = pts[3]; // bottom-left

        return ordered;
    }

    public Mat preprocessSudokuImage(String path) {
        Mat src = Imgcodecs.imread(path);
        Mat gray = new Mat();
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);

        // Equalize histogram for contrast improvement
        Imgproc.equalizeHist(gray, gray);

        // Blur to reduce noise
        Imgproc.GaussianBlur(gray, gray, new Size(7, 7), 0);

        Mat thresh = new Mat();
        // Adaptive Gaussian Threshold for better binarization
        Imgproc.adaptiveThreshold(gray, thresh, 255,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 11, 2);

        // Morphological opening to remove noise
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_OPEN, kernel);

        // Find contours to detect the Sudoku grid
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(thresh, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        double maxArea = 0;
        MatOfPoint maxContour = null;
        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > maxArea) {
                MatOfPoint2f approxCurve = new MatOfPoint2f();
                MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
                double peri = Imgproc.arcLength(contour2f, true);
                Imgproc.approxPolyDP(contour2f, approxCurve, 0.02 * peri, true);
                if (approxCurve.total() == 4) {
                    maxArea = area;
                    maxContour = new MatOfPoint(approxCurve.toArray());
                }
            }
        }

        if (maxContour == null)
            throw new RuntimeException("Could not find Sudoku grid");

        Point[] pts = maxContour.toArray();
        Point[] orderedPts = orderPoints(pts);

        double side = 450;
        MatOfPoint2f dest = new MatOfPoint2f(
                new Point(0, 0),
                new Point(side - 1, 0),
                new Point(side - 1, side - 1),
                new Point(0, side - 1));
        MatOfPoint2f srcPts = new MatOfPoint2f(orderedPts);
        Mat warpMat = Imgproc.getPerspectiveTransform(srcPts, dest);
        Mat warped = new Mat();
        Imgproc.warpPerspective(gray, warped, warpMat, new Size(side, side));
        return warped;
    }

    public List<Mat> splitCells(Mat warped) {
        int size = warped.rows();
        int cellSize = size / 9;
        int margin = cellSize / 10; // Crop margin inside each cell to avoid grid lines
        List<Mat> cells = new ArrayList<>();

        for (int row = 0; row < 9; row++) {
            for (int col = 0; col < 9; col++) {
                int x = col * cellSize + margin;
                int y = row * cellSize + margin;
                int width = cellSize - 2 * margin;
                int height = cellSize - 2 * margin;

                Rect roi = new Rect(x, y, width, height);
                Mat cell = new Mat(warped, roi);

                // Resize to standard size for OCR
                Mat resized = new Mat();
                Imgproc.resize(cell, resized, new Size(28, 28));

                cells.add(resized);
            }
        }
        return cells;
    }

    public int[][] recognizeDigits(List<Mat> cells, ITesseract tesseract) throws Exception {
        int[][] board = new int[9][9];
        tesseract.setTessVariable("tessedit_char_whitelist", "123456789");
        tesseract.setOcrEngineMode(ITessAPI.TessOcrEngineMode.OEM_LSTM_ONLY);
        tesseract.setPageSegMode(ITessAPI.TessPageSegMode.PSM_SINGLE_CHAR);

        for (int i = 0; i < cells.size(); i++) {
            BufferedImage img = matToBufferedImage(cells.get(i));
            String rawResult = tesseract.doOCR(img).trim();

            // Take only first recognized digit
            String digitStr = rawResult.replaceAll("[^1-9]", "");
            int digit = digitStr.isEmpty() ? 0 : Integer.parseInt(digitStr);

            board[i / 9][i % 9] = digit;
        }
        return board;
    }

    private BufferedImage matToBufferedImage(Mat mat) {
        int type = (mat.channels() > 1) ? BufferedImage.TYPE_3BYTE_BGR : BufferedImage.TYPE_BYTE_GRAY;
        int bufferSize = mat.channels() * mat.cols() * mat.rows();
        byte[] b = new byte[bufferSize];
        mat.get(0, 0, b);
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        final byte[] targetPixels = ((java.awt.image.DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(b, 0, targetPixels, 0, b.length);
        return image;
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
        OpenCV.loadLocally(); // This extracts native libs automatically
        SwingUtilities.invokeLater(SudokuSolverGUI::new);
    }
}
