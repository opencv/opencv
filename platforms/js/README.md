Building OpenCV.js by Emscripten
====================

[Download and install Emscripten](https://kripken.github.io/emscripten-site/docs/getting_started/downloads.html).

Execute `build_js.py` script:
```
python <opencv_src_dir>/platforms/js/build_js.py <build_dir>
```

If everything is fine, a few minutes later you will get `<build_dir>/bin/opencv.js`. You can add this into your web pages.

Find out more build options by `-h` switch.

For detailed build tutorial, check out `<opencv_src_dir>/doc/js_tutorials/js_setup/js_setup/js_setup.markdown`.
