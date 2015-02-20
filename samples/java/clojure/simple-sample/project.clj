(defproject simple-sample "0.1.0-SNAPSHOT"
  :pom-addition [:developers [:developer {:id "magomimmo"}
                              [:name "Mimmo Cosenza"]
                              [:url "https://github.com/magomimmoo"]]]

  :description "A simple project to start REPLing with OpenCV"
  :url "http://example.com/FIXME"
  :license {:name "BSD 3-Clause License"
            :url "http://opensource.org/licenses/BSD-3-Clause"}
  :dependencies [[org.clojure/clojure "1.5.1"]
                 [opencv/opencv "2.4.7"]
                 [opencv/opencv-native "2.4.7"]]
  :main simple-sample.core
  :injections [(clojure.lang.RT/loadLibrary org.opencv.core.Core/NATIVE_LIBRARY_NAME)])
