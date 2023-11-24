package org.opencv.samples.recorder;

import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.videoio.Videoio;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.util.Collections;
import java.util.List;

public class RecorderActivity extends CameraActivity implements CvCameraViewListener2, View.OnClickListener {
    private static final String TAG = "OCVSample::Activity";
    private static final String FILENAME_MP4 = "sample_video1.mp4";
    private static final String FILENAME_AVI = "sample_video1.avi";
    private String videoFilename;
    private boolean useMJPG = false;

    private static final int STATUS_FINISHED_PLAYBACK = 0;
    private static final int STATUS_PREVIEW = 1;
    private static final int STATUS_RECORDING = 2;
    private static final int STATUS_PLAYING = 3;
    private static final int STATUS_ERROR = 4;

    private int status = STATUS_FINISHED_PLAYBACK;
    private int FPS = 30;
    private int width = 0, height = 0;

    private CameraBridgeViewBase mOpenCvCameraView;
    private boolean mIsJavaCamera = true;
    private MenuItem mItemSwitchCamera = null;
    private ImageView imageView;
    private Button button;
    private TextView textView;
    Runnable runnable;

    private VideoWriter videoWriter = null;
    private VideoCapture videoCapture = null;

    public RecorderActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.recorder_surface_view);

        textView = (TextView) findViewById(R.id.textview1);
        textView.bringToFront();

        if (OpenCVLoader.initLocal()) {
            Log.i(TAG, "OpenCV loaded successfully");
        } else {
            Log.e(TAG, "OpenCV initialization failed!");
            status = STATUS_ERROR;
            textView.setText("Error: Can't initialize OpenCV");
            return;
        }

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.recorder_activity_java_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.GONE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.disableView();

        imageView = (ImageView) findViewById(R.id.image_view);

        button = (Button) findViewById(R.id.btn1);
        button.setOnClickListener(this);
        button.bringToFront();

        videoFilename = getFilesDir() + "/" + FILENAME_MP4;
    }

    @Override
    public void onPause()
    {
        Log.d(TAG, "Pause");
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
        imageView.setVisibility(SurfaceView.GONE);
        if (videoWriter != null) {
            videoWriter.release();
            videoWriter = null;
        }
        if (videoCapture != null) {
            videoCapture.release();
            videoCapture = null;
        }
        status = STATUS_FINISHED_PLAYBACK;
        textView.setText("Status: Finished playback");
        button.setText("Start Camera");
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    public void onDestroy() {
        Log.d(TAG, "called onDestroy");
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
        if (videoWriter != null)
            videoWriter.release();
        if (videoCapture != null)
            videoCapture.release();
    }

    public void onCameraViewStarted(int width, int height) {
        Log.d(TAG, "Camera view started " + String.valueOf(width) + "x" + String.valueOf(height));
        this.width = width;
        this.height = height;
    }

    public void onCameraViewStopped() {
        Log.d(TAG, "Camera view stopped");
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame)
    {
        Log.d(TAG, "Camera frame arrived");

        Mat rgbMat = inputFrame.rgba();
        Imgproc.cvtColor(rgbMat, rgbMat, Imgproc.COLOR_RGBA2RGB);

        int w = rgbMat.width();
        int h = rgbMat.height();

        Log.d(TAG, "Size: " + String.valueOf(w) + "x" + String.valueOf(h));

        if (videoWriter != null && videoWriter.isOpened()) {
            videoWriter.write(rgbMat);
        }

        return rgbMat;
    }

    @Override
    public void onClick(View view) {
        Log.i(TAG,"onClick event");
        ChangeStatus();
    }

    public void ChangeStatus() {
        switch(status) {
            case STATUS_ERROR:
                Toast.makeText(this, "Error", Toast.LENGTH_LONG).show();
                break;
            case STATUS_FINISHED_PLAYBACK:
                if (!startPreview()) {
                    setErrorStatus();
                    break;
                }
                status = STATUS_PREVIEW;
                textView.setText("Status: Camera preview");
                button.setText("Start recording");
                break;
            case STATUS_PREVIEW:
                if (!startRecording()) {
                    setErrorStatus();
                    break;
                }
                status = STATUS_RECORDING;
                textView.setText("Status: recording video");
                button.setText(" Stop and play video");
                break;
            case STATUS_RECORDING:
                if (!stopRecording()) {
                    setErrorStatus();
                    break;
                }
                if (!startPlayback()) {
                    setErrorStatus();
                    break;
                }
                status = STATUS_PLAYING;
                textView.setText("Status: Playing video");
                button.setText("Stop playback");
                break;
            case STATUS_PLAYING:
                if (!stopPlayback()) {
                    setErrorStatus();
                    break;
                }
                status = STATUS_FINISHED_PLAYBACK;
                textView.setText("Status: Finished playback");
                button.setText("Start Camera");
                break;
        }
    }

    public void setErrorStatus() {
        status = STATUS_ERROR;
        textView.setText("Status: Error");
    }

    public boolean startPreview() {
        mOpenCvCameraView.enableView();
        mOpenCvCameraView.setVisibility(View.VISIBLE);
        return true;
    }

    public boolean startRecording() {
        Log.i(TAG,"Starting recording");

        File file = new File(videoFilename);
        file.delete();

        videoWriter = new VideoWriter();
        if (!useMJPG) {
            videoWriter.open(videoFilename, Videoio.CAP_ANDROID, VideoWriter.fourcc('H', '2', '6', '4'), FPS, new Size(width, height));
            if (!videoWriter.isOpened()) {
                Log.i(TAG,"Can't record H264. Switching to MJPG");
                useMJPG = true;
                videoFilename = getFilesDir() + "/" + FILENAME_AVI;
            }
        }

        if (useMJPG) {
            videoWriter.open(videoFilename, VideoWriter.fourcc('M', 'J', 'P', 'G'), FPS, new Size(width, height));
        }

        Log.d(TAG, "Size: " + String.valueOf(width) + "x" + String.valueOf(height));
        Log.d(TAG, "File: " + videoFilename);

        if (videoWriter.isOpened()) {
            Toast.makeText(this, "Record started to file " + videoFilename, Toast.LENGTH_LONG).show();
            return true;
        } else {
            Toast.makeText(this, "Failed to start a record", Toast.LENGTH_LONG).show();
            return false;
        }
    }

    public boolean stopRecording() {
        Log.i(TAG, "Finishing recording");
        mOpenCvCameraView.disableView();
        mOpenCvCameraView.setVisibility(SurfaceView.GONE);
        videoWriter.release();
        videoWriter = null;
        return true;
    }

    public boolean startPlayback() {
        imageView.setVisibility(SurfaceView.VISIBLE);

        if (!useMJPG){
            videoCapture = new VideoCapture(videoFilename, Videoio.CAP_ANDROID);
        } else {
            videoCapture = new VideoCapture(videoFilename);
        }
        if (!videoCapture.isOpened()) {
            Log.e(TAG, "Can't open video");
            Toast.makeText(this, "Can't open file " + videoFilename, Toast.LENGTH_SHORT).show();
            return false;
        }
        Toast.makeText(this, "Starting playback from file " + videoFilename, Toast.LENGTH_SHORT).show();
        runnable = new Runnable() {
            @Override
            public void run() {
                if (videoCapture == null || !videoCapture.isOpened()) {
                    return;
                }
                Mat frame = new Mat();
                videoCapture.read(frame);
                if (frame.empty()) {
                    if (status == STATUS_PLAYING) {
                        ChangeStatus();
                    }
                    return;
                }
                if (!useMJPG)
                    Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2RGB);
                Bitmap bmp = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(frame, bmp);
                imageView.setImageBitmap(bmp);
                Handler h = new Handler();
                h.postDelayed(this, 33);
            }
        };

        runnable.run();
        return true;
    }

    public boolean stopPlayback() {
        videoCapture.release();
        videoCapture = null;
        imageView.setVisibility(SurfaceView.GONE);
        return true;
    }

}
