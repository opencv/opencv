## Model weights note

Model weights are not stored in this source code repository.

C++ samples download weights automatically (using CMake).
Python samples are able to download necessary files too.
You just need to have properly configured Internet connection.

Weights can be downloaded by running of download_weights.py script.

Alternatively, You can download them using Metalink (.meta4) descriptions
from 'download' directory:

* Install one of [tools supporting Metalink format](https://en.wikipedia.org/wiki/Metalink)

* Run this tool and pass `opencv_dnn_face_detector_uint8.meta4` file to it, e.g. `aria2c opencv_dnn_face_detector_uint8.meta4`
