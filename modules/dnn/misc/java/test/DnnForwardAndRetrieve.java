package org.opencv.test.dnn;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Range;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.test.OpenCVTestCase;

public class DnnForwardAndRetrieve extends OpenCVTestCase {

    public void testForwardAndRetrieve()
    {
        // Create a simple Caffe prototxt with a Slice layer
        String prototxt =
            "input: \"data\"\n" +
            "layer {\n" +
            "  name: \"testLayer\"\n" +
            "  type: \"Slice\"\n" +
            "  bottom: \"data\"\n" +
            "  top: \"firstCopy\"\n" +
            "  top: \"secondCopy\"\n" +
            "  slice_param {\n" +
            "    axis: 0\n" +
            "    slice_point: 2\n" +
            "  }\n" +
            "}";

        // Read network from prototxt
        MatOfByte bufferProto = new MatOfByte();
        bufferProto.fromArray(prototxt.getBytes());
        Net net = Dnn.readNetFromCaffe(bufferProto);
        net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);

        // Create input data
        Mat inp = new Mat(4, 5, CvType.CV_32F);
        Core.randu(inp, -1, 1);
        net.setInput(inp);

        // Define output names
        List<String> outNames = new ArrayList<>();
        outNames.add("testLayer");

        // Forward and retrieve multiple outputs
        List<List<Mat>> outBlobs = new ArrayList<>();
        net.forwardAndRetrieve(outBlobs, outNames);

        // Verify results
        assertEquals(1, outBlobs.size());
        assertEquals(2, outBlobs.get(0).size());

        // Compare results
        Mat expectedFirst = inp.rowRange(0, 2);
        Mat expectedSecond = inp.rowRange(2, 4);

        Mat actualFirst = outBlobs.get(0).get(0);
        Mat actualSecond = outBlobs.get(0).get(1);

        assertEquals(0, Core.norm(expectedFirst, actualFirst, Core.NORM_INF), EPS);
        assertEquals(0, Core.norm(expectedSecond, actualSecond, Core.NORM_INF), EPS);
    }
}
