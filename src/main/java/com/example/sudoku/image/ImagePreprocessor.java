package com.example.sudoku.image;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class ImagePreprocessor {

    public Mat preprocessSudokuImage(Mat src) {
        Mat gray = new Mat();
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);

        Imgproc.equalizeHist(gray, gray);
        Imgproc.GaussianBlur(gray, gray, new Size(7, 7), 0);

        Mat thresh = new Mat();
        Imgproc.adaptiveThreshold(gray, thresh, 255,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 11, 2);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_OPEN, kernel);

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
        if (maxContour == null) throw new RuntimeException("Could not find Sudoku grid");

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

    private Point[] orderPoints(Point[] pts) {
        Point[] ordered = new Point[4];
        Arrays.sort(pts, Comparator.comparingDouble(p -> p.x + p.y));
        ordered[0] = pts[0]; // top-left
        ordered[2] = pts[3]; // bottom-right

        Arrays.sort(pts, Comparator.comparingDouble(p -> p.y - p.x));
        ordered[1] = pts[0]; // top-right
        ordered[3] = pts[3]; // bottom-left

        return ordered;
    }
}
