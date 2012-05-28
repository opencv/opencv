android update project --target android-11 --library ../../OpenCV-2.4.0/ --name "Sample - 15-puzzle"                             --path ./15-puzzle
android update project --target android-11 --library ../../OpenCV-2.4.0/ --name "Sample - face-detection"                        --path ./face-detection
android update project --target android-11 --library ../../OpenCV-2.4.0/ --name "Sample - image-manipulations"                   --path ./image-manipulations
android update project --target android-11                               --name "Tutorial 0 (Basic) - Android Camera"            --path ./tutorial-0-androidcamera
android update project --target android-11 --library ../../OpenCV-2.4.0/ --name "Tutorial 1 (Basic) - Add OpenCV"                --path ./tutorial-1-addopencv
android update project --target android-11 --library ../../OpenCV-2.4.0/ --name "Tutorial 2 (Basic) - Use OpenCV Camera"         --path ./tutorial-2-opencvcamera
android update project --target android-11                               --name "Tutorial 3 (Advanced) - Add Native OpenCV"      --path ./tutorial-3-native
android update project --target android-11 --library ../../OpenCV-2.4.0/ --name "Tutorial 4 (Advanced) - Mix Java+Native OpenCV" --path ./tutorial-4-mixed

exit

rm ./15-puzzle/local.properties
rm ./face-detection/local.properties
rm ./image-manipulations/local.properties
rm ./tutorial-0-androidcamera/local.properties
rm ./tutorial-1-addopencv/local.properties
rm ./tutorial-2-opencvcamera/local.properties
rm ./tutorial-3-native/local.properties
rm ./tutorial-4-mixed/local.properties