package org.opencv.samples.puzzle15;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;

import android.util.Log;


/**
 * This class is a controller for puzzle game.
 * It converts the image from Camera into the shuffled image
 */
public class Puzzle15Processor {

    private static final int GRID_SIZE = 4;
    private static final int GRID_AREA = GRID_SIZE * GRID_SIZE;
    private static final int GRID_EMPTY_INDEX = GRID_AREA - 1;
    private static final String TAG = "Puzzle15Processor";
    private static final Scalar GRID_EMPTY_COLOR = new Scalar(0x33, 0x33, 0x33, 0xFF);

    private int[]   mIndexes;
    private int[]   mTextWidths;
    private int[]   mTextHeights;

    private Mat mRgba15;
    private Mat[] mCells15;
    private boolean mShowTileNumbers = true;

    public Puzzle15Processor() {
        mTextWidths = new int[GRID_AREA];
        mTextHeights = new int[GRID_AREA];

        mIndexes = new int [GRID_AREA];

        for (int i = 0; i < GRID_AREA; i++)
            mIndexes[i] = i;
    }

    /* this method is intended to make processor prepared for a new game */
    public synchronized void prepareNewGame() {
        do {
            shuffle(mIndexes);
        } while (!isPuzzleSolvable());
    }

    /* This method is to make the processor know the size of the frames that
     * will be delivered via puzzleFrame.
     * If the frames will be different size - then the result is unpredictable
     */
    public synchronized void prepareGameSize(int width, int height) {
        mRgba15 = new Mat(height, width, CvType.CV_8UC4);
        mCells15 = new Mat[GRID_AREA];

        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                int k = i * GRID_SIZE + j;
                mCells15[k] = mRgba15.submat(i * height / GRID_SIZE, (i + 1) * height / GRID_SIZE, j * width / GRID_SIZE, (j + 1) * width / GRID_SIZE);
            }
        }

        for (int i = 0; i < GRID_AREA; i++) {
            Size s = Imgproc.getTextSize(Integer.toString(i + 1), Imgproc.FONT_HERSHEY_COMPLEX, 1, 2, null);
            mTextHeights[i] = (int) s.height;
            mTextWidths[i] = (int) s.width;
        }
    }

    /* this method to be called from the outside. it processes the frame and shuffles
     * the tiles as specified by mIndexes array
     */
    public synchronized Mat puzzleFrame(Mat inputPicture) {
        Mat[] cells = new Mat[GRID_AREA];
        int rows = inputPicture.rows();
        int cols = inputPicture.cols();

        rows = rows - rows%4;
        cols = cols - cols%4;

        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                int k = i * GRID_SIZE + j;
                cells[k] = inputPicture.submat(i * inputPicture.rows() / GRID_SIZE, (i + 1) * inputPicture.rows() / GRID_SIZE, j * inputPicture.cols()/ GRID_SIZE, (j + 1) * inputPicture.cols() / GRID_SIZE);
            }
        }

        rows = rows - rows%4;
        cols = cols - cols%4;

        // copy shuffled tiles
        for (int i = 0; i < GRID_AREA; i++) {
            int idx = mIndexes[i];
            if (idx == GRID_EMPTY_INDEX)
                mCells15[i].setTo(GRID_EMPTY_COLOR);
            else {
                cells[idx].copyTo(mCells15[i]);
                if (mShowTileNumbers) {
                    Imgproc.putText(mCells15[i], Integer.toString(1 + idx), new Point((cols / GRID_SIZE - mTextWidths[idx]) / 2,
                            (rows / GRID_SIZE + mTextHeights[idx]) / 2), Imgproc.FONT_HERSHEY_COMPLEX, 1, new Scalar(255, 0, 0, 255), 2);
                }
            }
        }

        for (int i = 0; i < GRID_AREA; i++)
            cells[i].release();

        drawGrid(cols, rows, mRgba15);

        return mRgba15;
    }

    public void toggleTileNumbers() {
        mShowTileNumbers = !mShowTileNumbers;
    }

    public void deliverTouchEvent(int x, int y) {
        int rows = mRgba15.rows();
        int cols = mRgba15.cols();

        int row = (int) Math.floor(y * GRID_SIZE / rows);
        int col = (int) Math.floor(x * GRID_SIZE / cols);

        if (row < 0 || row >= GRID_SIZE || col < 0 || col >= GRID_SIZE) {
            Log.e(TAG, "It is not expected to get touch event outside of picture");
            return ;
        }

        int idx = row * GRID_SIZE + col;
        int idxtoswap = -1;

        // left
        if (idxtoswap < 0 && col > 0)
            if (mIndexes[idx - 1] == GRID_EMPTY_INDEX)
                idxtoswap = idx - 1;
        // right
        if (idxtoswap < 0 && col < GRID_SIZE - 1)
            if (mIndexes[idx + 1] == GRID_EMPTY_INDEX)
                idxtoswap = idx + 1;
        // top
        if (idxtoswap < 0 && row > 0)
            if (mIndexes[idx - GRID_SIZE] == GRID_EMPTY_INDEX)
                idxtoswap = idx - GRID_SIZE;
        // bottom
        if (idxtoswap < 0 && row < GRID_SIZE - 1)
            if (mIndexes[idx + GRID_SIZE] == GRID_EMPTY_INDEX)
                idxtoswap = idx + GRID_SIZE;

        // swap
        if (idxtoswap >= 0) {
            synchronized (this) {
                int touched = mIndexes[idx];
                mIndexes[idx] = mIndexes[idxtoswap];
                mIndexes[idxtoswap] = touched;
            }
        }
    }

    private void drawGrid(int cols, int rows, Mat drawMat) {
        for (int i = 1; i < GRID_SIZE; i++) {
            Imgproc.line(drawMat, new Point(0, i * rows / GRID_SIZE), new Point(cols, i * rows / GRID_SIZE), new Scalar(0, 255, 0, 255), 3);
            Imgproc.line(drawMat, new Point(i * cols / GRID_SIZE, 0), new Point(i * cols / GRID_SIZE, rows), new Scalar(0, 255, 0, 255), 3);
        }
    }

    private static void shuffle(int[] array) {
        for (int i = array.length; i > 1; i--) {
            int temp = array[i - 1];
            int randIx = (int) (Math.random() * i);
            array[i - 1] = array[randIx];
            array[randIx] = temp;
        }
    }

    private boolean isPuzzleSolvable() {

        int sum = 0;
        for (int i = 0; i < GRID_AREA; i++) {
            if (mIndexes[i] == GRID_EMPTY_INDEX)
                sum += (i / GRID_SIZE) + 1;
            else {
                int smaller = 0;
                for (int j = i + 1; j < GRID_AREA; j++) {
                    if (mIndexes[j] < mIndexes[i])
                        smaller++;
                }
                sum += smaller;
            }
        }
        return sum % 2 == 0;
    }
}
