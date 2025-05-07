package org.opencv.test.dnn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.Range;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Image2BlobParams;
import org.opencv.test.OpenCVTestCase;

public class DnnBlobFromImageWithParamsTest extends OpenCVTestCase {

        public void testBlobFromImageWithParamsNHWCScalarScale()
        {
            // https://github.com/opencv/opencv/issues/27264
            /*
            Mat img = new Mat(10, 10, CvType.CV_8UC4, new Scalar(0, 1, 2, 3));
            Scalar scalefactor = new Scalar(0.1, 0.2, 0.3, 0.4);

            Image2BlobParams params = new Image2BlobParams();
            params.set_scalefactor(scalefactor);
            params.set_datalayout(Core.DATA_LAYOUT_NHWC);
            return;

            Mat blob = Dnn.blobFromImageWithParams(img, params); // [1, 10, 10, 4]

            float[] expectedValues = { (float)scalefactor.val[0] * 0, (float)scalefactor.val[1] * 1, (float)scalefactor.val[2] * 2, (float)scalefactor.val[3] * 3 }; // Target Value.
            for (int h = 0; h < 10; h++)
            {
                for (int w = 0; w < 10; w++)
                {
                      float[] actualValues = new float[4];
                      blob.get(new int[]{0, h, w, 0}, actualValues);
                      for (int c = 0; c < 4; c++)
                      {
                          // Check equal
                          assertEquals(expectedValues[c], actualValues[c]);
                      }
                }
            }
            */
        }

        public void testBlobFromImageWithParamsCustomPaddingLetterBox()
        {
            Mat img = new Mat(40, 20, CvType.CV_8UC4, new Scalar(0, 1, 2, 3));

            // Custom padding value that you have added
            Scalar customPaddingValue = new Scalar(5, 6, 7, 8); // Example padding value
            Size targetSize = new Size(20, 20);

            Mat targetImg = img.clone();
            Core.copyMakeBorder(targetImg, targetImg, 0, 0, (int)targetSize.width / 2, (int)targetSize.width / 2, Core.BORDER_CONSTANT, customPaddingValue);

            // Set up Image2BlobParams with your new functionality
            Image2BlobParams params = new Image2BlobParams();
            params.set_size(targetSize);
            params.set_paddingmode(Dnn.DNN_PMODE_LETTERBOX);
            params.set_borderValue(customPaddingValue); // Use your new feature here

            // Create blob with custom padding
            Mat blob = Dnn.blobFromImageWithParams(img, params);

            // Create target blob for comparison
            Mat targetBlob = Dnn.blobFromImage(targetImg, 1.0, targetSize);

            assertEquals(0, Core.norm(targetBlob, blob, Core.NORM_INF), EPS);
        }

        public void testBlobFromImageWithParams4chLetterBox()
        {
            Mat img = new Mat(40, 20, CvType.CV_8UC4, new Scalar(0, 1, 2, 3));

            // Construct target mat.
            Mat[] targetChannels = new Mat[4];

            // The letterbox will add zero at the left and right of output blob.
            // After the letterbox, every row data would have same value showing as valVec.
            byte[] valVec = { 0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0};

            Mat rowM = new Mat(1, 20, CvType.CV_8UC1);
            rowM.put(0, 0, valVec);
            for (int i = 0; i < 4; i++) {
                Core.multiply(rowM, new Scalar(i), targetChannels[i] = new Mat());
            }

            Mat targetImg = new Mat();
            Core.merge(Arrays.asList(targetChannels), targetImg);
            Size targetSize = new Size(20, 20);

            Image2BlobParams params = new Image2BlobParams();
            params.set_size(targetSize);
            params.set_paddingmode(Dnn.DNN_PMODE_LETTERBOX);
            Mat blob = Dnn.blobFromImageWithParams(img, params);
            Mat targetBlob = Dnn.blobFromImage(targetImg, 1.0, targetSize); // only convert data from uint8 to float32.

            assertEquals(0, Core.norm(targetBlob, blob, Core.NORM_INF), EPS);
        }

        // https://github.com/opencv/opencv/issues/27264
        public void testBlobFromImageWithParams4chMultiImage()
        {
            /*
            Mat img = new Mat(10, 10, CvType.CV_8UC4, new Scalar(0, 1, 2, 3));

            Scalar scalefactor = new Scalar(0.1, 0.2, 0.3, 0.4);

            Image2BlobParams param = new Image2BlobParams();
            param.set_scalefactor(scalefactor);
            param.set_datalayout(Core.DATA_LAYOUT_NHWC);
            return;

            List<Mat> images = new ArrayList<>();
            images.add(img);
            Mat img2 = new Mat();
            Core.multiply(img, Scalar.all(2), img2);
            images.add(img2);

            Mat blobs = Dnn.blobFromImagesWithParams(images, param);

            Range[] ranges = new Range[4];
            ranges[0] = new Range(0, 1);
            ranges[1] = new Range(0, blobs.size(1));
            ranges[2] = new Range(0, blobs.size(2));
            ranges[3] = new Range(0, blobs.size(3));

            Mat blob0 = blobs.submat(ranges).clone();

            ranges[0] = new Range(1, 2);
            Mat blob1 = blobs.submat(ranges).clone();

            Core.multiply(blob0, Scalar.all(2), blob0);

            assertEquals(0, Core.norm(blob0, blob1, Core.NORM_INF), EPS);
            */
        }
}
