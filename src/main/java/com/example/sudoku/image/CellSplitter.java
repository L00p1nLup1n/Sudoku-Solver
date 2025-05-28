package com.example.sudoku.image;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class CellSplitter {

    public List<Mat> splitCells(Mat warped) {
        int size = warped.rows();
        int cellSize = size / 9;
        int margin = cellSize / 10; // crop margin inside cell
        List<Mat> cells = new ArrayList<>();

        for (int row = 0; row < 9; row++) {
            for (int col = 0; col < 9; col++) {
                int x = col * cellSize + margin;
                int y = row * cellSize + margin;
                int width = cellSize - 2 * margin;
                int height = cellSize - 2 * margin;
                Rect roi = new Rect(x, y, width, height);
                Mat cell = new Mat(warped, roi);
                Mat resized = new Mat();
                Imgproc.resize(cell, resized, new Size(28, 28));
                cells.add(resized);
            }
        }
        return cells;
    }
}
