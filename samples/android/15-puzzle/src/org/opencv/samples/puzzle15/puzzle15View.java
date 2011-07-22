package org.opencv.samples.puzzle15;

import org.opencv.Android;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;

import android.content.Context;
import android.graphics.Bitmap;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.View;
import android.view.View.OnTouchListener;

public class puzzle15View extends SampleCvViewBase implements OnTouchListener {
    private Mat     mRgba;
    private Mat     mRgba15;
    private Mat[]   mCells;
    private Mat[]   mCells15;
    private int[]   mIndexses;
    private int[]   mTextWidths;
    private int[]   mTextHeights;
    private boolean mShowTileNumbers = true;

    int             gridSize         = 4;
    int             gridArea         = gridSize * gridSize;
    int             gridEmptyIdx     = gridArea - 1;

    public puzzle15View(Context context) {
        super(context);
        setOnTouchListener(this);

        mTextWidths = new int[gridArea];
        mTextHeights = new int[gridArea];
        for (int i = 0; i < gridArea; i++) {
            Size s = Core.getTextSize(Integer.toString(i + 1), 3/* CV_FONT_HERSHEY_COMPLEX */, 1, 2, null);
            mTextHeights[i] = (int) s.height;
            mTextWidths[i] = (int) s.width;
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder _holder, int format, int width, int height) {
        super.surfaceChanged(_holder, format, width, height);
        synchronized (this) {
            // initialize Mat before usage
            mRgba = new Mat();
        }
    }

    public static void shuffle(int[] array) {
        for (int i = array.length; i > 1; i--) {
            int temp = array[i - 1];
            int randIx = (int) (Math.random() * i);
            array[i - 1] = array[randIx];
            array[randIx] = temp;
        }
    }

    public boolean isPuzzleSolvable() {
        if (gridSize != 4)
            return true;

        int sum = 0;
        for (int i = 0; i < gridArea; i++) {
            if (mIndexses[i] == gridEmptyIdx)
                sum += (i / gridSize) + 1;
            else {
                int smaller = 0;
                for (int j = i + 1; j < gridArea; j++) {
                    if (mIndexses[j] < mIndexses[i])
                        smaller++;
                }
                sum += smaller;
            }
        }

        return sum % 2 == 0;
    }

    private void createPuzzle(int cols, int rows) {
        mCells = new Mat[gridArea];
        mCells15 = new Mat[gridArea];

        mRgba15 = new Mat(rows, cols, mRgba.type());
        mIndexses = new int[gridArea];

        for (int i = 0; i < gridSize; i++) {
            for (int j = 0; j < gridSize; j++) {
                int k = i * gridSize + j;
                mIndexses[k] = k;
                mCells[k] = mRgba.submat(i * rows / gridSize, (i + 1) * rows / gridSize, j * cols / gridSize, (j + 1) * cols / gridSize);
                mCells15[k] = mRgba15.submat(i * rows / gridSize, (i + 1) * rows / gridSize, j * cols / gridSize, (j + 1) * cols / gridSize);
            }
        }

        startNewGame();
    }

    public void startNewGame() {
        do {
            shuffle(mIndexses);
        } while (!isPuzzleSolvable());
    }

    public void tolggleTileNumbers() {
        mShowTileNumbers = !mShowTileNumbers;
    }

    @Override
    protected Bitmap processFrame(VideoCapture capture) {
        capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
        int cols = mRgba.cols();
        int rows = mRgba.rows();

        if (mCells == null)
            createPuzzle(cols, rows);

        // copy shuffled tiles
        for (int i = 0; i < gridArea; i++) {
            int idx = mIndexses[i];
            if (idx == gridEmptyIdx)
                mCells15[i].setTo(new Scalar(0x33, 0x33, 0x33, 0xFF));
            else {
                mCells[idx].copyTo(mCells15[i]);
                if (mShowTileNumbers) {
                    Core.putText(mCells15[i], Integer.toString(1 + idx), new Point((cols / gridSize - mTextWidths[idx]) / 2,
                            (rows / gridSize + mTextHeights[idx]) / 2), 3/* CV_FONT_HERSHEY_COMPLEX */, 1, new Scalar(255, 0, 0, 255), 2);
                }
            }
        }

        drawGrid(cols, rows);

        Bitmap bmp = Bitmap.createBitmap(cols, rows, Bitmap.Config.ARGB_8888);
        if (Android.MatToBitmap(mRgba15, bmp))
            return bmp;

        bmp.recycle();
        return null;
    }

    private void drawGrid(int cols, int rows) {
        for (int i = 1; i < gridSize; i++) {
            Core.line(mRgba15, new Point(0, i * rows / gridSize), new Point(cols, i * rows / gridSize), new Scalar(0, 255, 0, 255), 3);
            Core.line(mRgba15, new Point(i * cols / gridSize, 0), new Point(i * cols / gridSize, rows), new Scalar(0, 255, 0, 255), 3);
        }
    }

    @Override
    public void run() {
        super.run();

        synchronized (this) {
            // Explicitly deallocate Mats
            if (mCells != null) {
                for (Mat m : mCells)
                    m.dispose();
            }
            if (mCells15 != null) {
                for (Mat m : mCells15)
                    m.dispose();
            }
            if (mRgba != null)
                mRgba.dispose();
            if (mRgba15 != null)
                mRgba15.dispose();

            mRgba = null;
            mRgba15 = null;
            mCells = null;
            mCells15 = null;
            mIndexses = null;
        }
    }

    public boolean onTouch(View v, MotionEvent event) {
        int cols = mRgba.cols();
        int rows = mRgba.rows();
        float xoffset = (getWidth() - cols) / 2;
        float yoffset = (getHeight() - rows) / 2;

        float x = event.getX() - xoffset;
        float y = event.getY() - yoffset;

        int row = (int) Math.floor(y * gridSize / rows);
        int col = (int) Math.floor(x * gridSize / cols);

        if (row < 0 || row >= gridSize || col < 0 || col >= gridSize)
            return false;

        int idx = row * gridSize + col;
        int idxtoswap = -1;

        // left
        if (idxtoswap < 0 && col > 0)
            if (mIndexses[idx - 1] == gridEmptyIdx)
                idxtoswap = idx - 1;
        // right
        if (idxtoswap < 0 && col < gridSize - 1)
            if (mIndexses[idx + 1] == gridEmptyIdx)
                idxtoswap = idx + 1;
        // top
        if (idxtoswap < 0 && row > 0)
            if (mIndexses[idx - gridSize] == gridEmptyIdx)
                idxtoswap = idx - gridSize;
        // bottom
        if (idxtoswap < 0 && row < gridSize - 1)
            if (mIndexses[idx + gridSize] == gridEmptyIdx)
                idxtoswap = idx + gridSize;

        // swap
        if (idxtoswap >= 0) {
            synchronized (this) {
                int touched = mIndexses[idx];
                mIndexses[idx] = mIndexses[idxtoswap];
                mIndexses[idxtoswap] = touched;
            }
        }

        return false;// don't need subsequent touch events
    }
}
