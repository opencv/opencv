;;; to run this code from the terminal: "$ lein run". It will save a
;;; blurred image version of resources/images/lena.png as
;;; resources/images/blurred.png

(ns simple-sample.core
  (:import [org.opencv.core Point Rect Mat CvType Size Scalar]
           org.opencv.highgui.Highgui
           org.opencv.imgproc.Imgproc))

(defn -main [& args]
  (let [lena (Highgui/imread "resources/images/lena.png")
        blurred (Mat. 512 512 CvType/CV_8UC3)]
    (print "Blurring...")
    (Imgproc/GaussianBlur lena blurred (Size. 5 5) 3 3)
    (Highgui/imwrite "resources/images/blurred.png" blurred)
    (println "done!")))
