import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.Aruco2;
import org.opencv.objdetect.Aruco2_GridBoard;
import org.opencv.objdetect.DetectionParameters;
import org.opencv.objdetect.FiducialMarker;

import java.util.List;

class Aruco2DetectionRun {
    public void run() {
        //! [generate_marker]
        Mat markerImg = new Mat();
        Aruco2.getFiducialMarker(markerImg, Aruco2.DICT_ARUCO_MIP_36h12, 42, 20, true);
        //! [generate_marker]

        System.out.println("Generated marker size: " + markerImg.cols() + "x" + markerImg.rows());

        // Create a scene with the marker on a white background
        Mat scene = new Mat(300, 300, CvType.CV_8UC1, new Scalar(255));
        markerImg.copyTo(scene.submat(50, 50 + markerImg.rows(), 50, 50 + markerImg.cols()));

        //! [detect_single]
        List<FiducialMarker> markers = Aruco2.detectFiducialMarkers(scene, Aruco2.DICT_ARUCO_MIP_36h12);
        //! [detect_single]

        System.out.println("Detected " + markers.size() + " marker(s)");
        for (FiducialMarker m : markers) {
            System.out.println("  Marker ID: " + m.get_id());
        }

        // Draw detected markers for visualization
        Mat colorScene = new Mat();
        Imgproc.cvtColor(scene, colorScene, Imgproc.COLOR_GRAY2BGR);

        //! [draw_markers]
        Aruco2.drawFiducialMarkers(colorScene, markers);
        //! [draw_markers]

        // Multi-dictionary detection
        //! [multi_dict]
        MatOfInt dicts = new MatOfInt(Aruco2.DICT_ARUCO_MIP_36h12, Aruco2.DICT_APRILTAG_36h11);
        List<FiducialMarker> multiMarkers = Aruco2.detectFiducialMarkers(scene, dicts, new DetectionParameters());
        //! [multi_dict]

        System.out.println("Multi-dictionary detection found " + multiMarkers.size() + " marker(s)");

        // Detection parameters
        //! [params]
        DetectionParameters params = new DetectionParameters();
        params.set_boxFilterSize(15);
        params.set_thres(3);
        params.set_errorCorrectionRate(0.0);

        List<FiducialMarker> paramsMarkers = Aruco2.detectFiducialMarkers(scene, Aruco2.DICT_ARUCO_MIP_36h12, params);
        //! [params]

        System.out.println("Custom parameters detection found " + paramsMarkers.size() + " marker(s)");

        // Grid board detection
        //! [grid_board]
        Mat boardImg = new Mat();
        Size gridSize = new Size(3, 2);
        Aruco2.getGridBoard(boardImg, gridSize, Aruco2.DICT_ARUCO_MIP_36h12, 20);

        Mat boardScene = new Mat(boardImg.rows() + 100, boardImg.cols() + 100, CvType.CV_8UC1, new Scalar(255));
        boardImg.copyTo(boardScene.submat(50, 50 + boardImg.rows(), 50, 50 + boardImg.cols()));

        Aruco2_GridBoard board = new Aruco2_GridBoard();
        boolean found = Aruco2.detectGridBoard(boardScene, gridSize, Aruco2.DICT_ARUCO_MIP_36h12, board);
        //! [grid_board]

        if (found) {
            System.out.println("Board detected with " + board.get_markers().size() + " markers");
        }
    }
}

public class Aruco2Detection {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new Aruco2DetectionRun().run();
    }
}
