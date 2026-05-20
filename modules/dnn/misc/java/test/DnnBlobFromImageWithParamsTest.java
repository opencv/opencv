package org.opencv.test.dnn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Range;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Image2BlobParams;
import org.opencv.test.OpenCVTestCase;

public class DnnBlobFromImageWithParamsTest extends OpenCVTestCase {

    // Verifies that DATA_LAYOUT_* and DNN_LAYOUT_* constants are accessible from Core
    // and carry the correct integer values. Exercises the missing_consts fix in
    // modules/core/misc/java/gen_dict.json (issue #27264).
    public void testDataLayoutConstants()
    {
        assertEquals(0, Core.DATA_LAYOUT_UNKNOWN);
        assertEquals(1, Core.DATA_LAYOUT_ND);
        assertEquals(2, Core.DATA_LAYOUT_NCHW);
        assertEquals(3, Core.DATA_LAYOUT_NCDHW);
        assertEquals(4, Core.DATA_LAYOUT_NHWC);
        assertEquals(5, Core.DATA_LAYOUT_NDHWC);
        assertEquals(6, Core.DATA_LAYOUT_PLANAR);
        assertEquals(7, Core.DATA_LAYOUT_BLOCK);

        // DNN_LAYOUT_* aliases must match the primary names
        assertEquals(Core.DATA_LAYOUT_UNKNOWN, Core.DNN_LAYOUT_UNKNOWN);
        assertEquals(Core.DATA_LAYOUT_ND,      Core.DNN_LAYOUT_ND);
        assertEquals(Core.DATA_LAYOUT_NCHW,    Core.DNN_LAYOUT_NCHW);
        assertEquals(Core.DATA_LAYOUT_NCDHW,   Core.DNN_LAYOUT_NCDHW);
        assertEquals(Core.DATA_LAYOUT_NHWC,    Core.DNN_LAYOUT_NHWC);
        assertEquals(Core.DATA_LAYOUT_NDHWC,   Core.DNN_LAYOUT_NDHWC);
        assertEquals(Core.DATA_LAYOUT_PLANAR,  Core.DNN_LAYOUT_PLANAR);
        assertEquals(Core.DATA_LAYOUT_BLOCK,   Core.DNN_LAYOUT_BLOCK);
    }

    // C++ reference: modules/dnn/test/test_misc.cpp — blobFromImageWithParams_4ch/NHWC_scalar_scale
    public void testBlobFromImageWithParamsNHWCScalarScale()
    {
        Mat img = new Mat(10, 10, CvType.CV_8UC4, new Scalar(0, 1, 2, 3));
        Scalar scalefactor = new Scalar(0.1, 0.2, 0.3, 0.4);

        Image2BlobParams params = new Image2BlobParams();
        params.set_scalefactor(scalefactor);
        params.set_datalayout(Core.DATA_LAYOUT_NHWC);

        Mat blob = Dnn.blobFromImageWithParams(img, params); // shape [1, 10, 10, 4]

        float[] expected = {
            (float)(scalefactor.val[0] * 0),
            (float)(scalefactor.val[1] * 1),
            (float)(scalefactor.val[2] * 2),
            (float)(scalefactor.val[3] * 3)
        };

        for (int h = 0; h < 10; h++) {
            for (int w = 0; w < 10; w++) {
                float[] actual = new float[4];
                blob.get(new int[]{0, h, w, 0}, actual);
                for (int c = 0; c < 4; c++) {
                    assertEquals(expected[c], actual[c], 1e-5f);
                }
            }
        }
    }

    // C++ reference: modules/dnn/test/test_misc.cpp — blobFromImageWithParams_CustomPadding/letter_box
    public void testBlobFromImageWithParamsCustomPaddingLetterBox()
    {
        Mat img = new Mat(40, 20, CvType.CV_8UC4, new Scalar(0, 1, 2, 3));

        Scalar customPaddingValue = new Scalar(5, 6, 7, 8);
        Size targetSize = new Size(20, 20);

        Mat targetImg = img.clone();
        Core.copyMakeBorder(targetImg, targetImg, 0, 0,
                (int)targetSize.width / 2, (int)targetSize.width / 2,
                Core.BORDER_CONSTANT, customPaddingValue);

        Image2BlobParams params = new Image2BlobParams();
        params.set_size(targetSize);
        params.set_paddingmode(Dnn.DNN_PMODE_LETTERBOX);
        params.set_borderValue(customPaddingValue);

        Mat blob       = Dnn.blobFromImageWithParams(img, params);
        Mat targetBlob = Dnn.blobFromImage(targetImg, 1.0, targetSize);

        assertEquals(0.0, Core.norm(targetBlob, blob, Core.NORM_INF), EPS);
    }

    // C++ reference: modules/dnn/test/test_misc.cpp — blobFromImageWithParams_4ch/letter_box
    public void testBlobFromImageWithParams4chLetterBox()
    {
        Mat img = new Mat(40, 20, CvType.CV_8UC4, new Scalar(0, 1, 2, 3));

        // Letterbox pads left/right with zeros; each row becomes valVec for each channel.
        byte[] valVec = { 0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0 };
        Mat rowM = new Mat(1, 20, CvType.CV_8UC1);
        rowM.put(0, 0, valVec);

        Mat[] targetChannels = new Mat[4];
        for (int i = 0; i < 4; i++) {
            Core.multiply(rowM, new Scalar(i), targetChannels[i] = new Mat());
        }

        Mat targetImg = new Mat();
        Core.merge(Arrays.asList(targetChannels), targetImg);
        Size targetSize = new Size(20, 20);

        Image2BlobParams params = new Image2BlobParams();
        params.set_size(targetSize);
        params.set_paddingmode(Dnn.DNN_PMODE_LETTERBOX);

        Mat blob       = Dnn.blobFromImageWithParams(img, params);
        Mat targetBlob = Dnn.blobFromImage(targetImg, 1.0, targetSize);

        assertEquals(0.0, Core.norm(targetBlob, blob, Core.NORM_INF), EPS);
    }

    // C++ reference: modules/dnn/test/test_misc.cpp — blobFromImagesWithParams_4ch/multi_image
    public void testBlobFromImageWithParams4chMultiImage()
    {
        Mat img = new Mat(10, 10, CvType.CV_8UC4, new Scalar(0, 1, 2, 3));
        Scalar scalefactor = new Scalar(0.1, 0.2, 0.3, 0.4);

        Image2BlobParams params = new Image2BlobParams();
        params.set_scalefactor(scalefactor);
        params.set_datalayout(Core.DATA_LAYOUT_NHWC);

        Mat img2 = new Mat();
        Core.multiply(img, Scalar.all(2), img2);

        List<Mat> images = new ArrayList<>();
        images.add(img);
        images.add(img2);

        Mat blobs = Dnn.blobFromImagesWithParams(images, params); // shape [2, 10, 10, 4]

        Range[] ranges = new Range[4];
        ranges[0] = new Range(0, 1);
        ranges[1] = new Range(0, blobs.size(1));
        ranges[2] = new Range(0, blobs.size(2));
        ranges[3] = new Range(0, blobs.size(3));
        Mat blob0 = blobs.submat(ranges).clone();

        ranges[0] = new Range(1, 2);
        Mat blob1 = blobs.submat(ranges).clone();

        // img2 = 2 * img, so blob1 must equal 2 * blob0
        Core.multiply(blob0, Scalar.all(2), blob0);
        assertEquals(0.0, Core.norm(blob0, blob1, Core.NORM_INF), 1e-5);
    }
}
