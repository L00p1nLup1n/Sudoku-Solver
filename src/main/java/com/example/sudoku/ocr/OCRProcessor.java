package com.example.sudoku.ocr;

import java.awt.image.BufferedImage;
import java.util.List;

import org.opencv.core.Mat;

import net.sourceforge.tess4j.ITessAPI;
import net.sourceforge.tess4j.ITesseract;

public class OCRProcessor {

    public int[][] recognizeDigits(List<Mat> cellImages, ITesseract tesseract) throws Exception {
        int[][] board = new int[9][9];
        
        tesseract.setDatapath("tessdata");
        tesseract.setVariable("tessedit_char_whitelist", "123456789");
        tesseract.setOcrEngineMode(ITessAPI.TessOcrEngineMode.OEM_LSTM_ONLY);
        tesseract.setPageSegMode(ITessAPI.TessPageSegMode.PSM_SINGLE_CHAR);

        for (int i = 0; i < cellImages.size(); i++) {
            BufferedImage img = ImageConverter.matToBufferedImage(cellImages.get(i));
            String rawResult = tesseract.doOCR(img).trim();
            String digitStr = rawResult.replaceAll("[^1-9]", "");
            int digit = digitStr.isEmpty() ? 0 : Integer.parseInt(digitStr);
            board[i / 9][i % 9] = digit;
        }
        return board;
    }
}
