# Use this script to generate test data for dnn module and TFLite models
import os
import numpy as np
import tensorflow as tf
import mediapipe as mp

import cv2 as cv

testdata = os.environ['OPENCV_TEST_DATA_PATH']

image = cv.imread(os.path.join(testdata, "cv", "shared", "lena.png"))
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

def run_tflite_model(model_name, inp_size):
    interpreter = tf.lite.Interpreter(model_name + ".tflite",
                                      experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Run model
    inp = cv.resize(image, inp_size)
    inp = np.expand_dims(inp, 0)
    inp = inp.astype(np.float32) / 255  # NHWC

    interpreter.set_tensor(input_details[0]['index'], inp)

    interpreter.invoke()

    for details in output_details:
        out = interpreter.get_tensor(details['index'])  # Or use an intermediate layer index
        out_name = details['name']
        np.save(f"{model_name}_out_{out_name}.npy", out)


def run_mediapipe_solution(solution, inp_size):
    with solution as selfie_segmentation:
        inp = cv.resize(image, inp_size)
        results = selfie_segmentation.process(inp)
        np.save(f"selfie_segmentation_out_activation_10.npy", results.segmentation_mask)

run_tflite_model("face_landmark", (192, 192))
run_tflite_model("face_detection_short_range", (128, 128))

run_mediapipe_solution(mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0), (256, 256))

# Save TensorFlow model as TFLite
def save_tflite_model(model, inp, name):
    func = model.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([func])
    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)

    with open(f'{name}.tflite', 'wb') as f:
        f.write(tflite_model)

    out = model(inp)

    np.save(f'{name}_inp.npy', inp.transpose(0, 3, 1, 2))
    np.save(f'{name}_out_Identity.npy', np.array(out).transpose(0, 3, 1, 2))


@tf.function(input_signature=[tf.TensorSpec(shape=[1, 3, 3, 1], dtype=tf.float32)])
def replicate_by_pack(x):
    pack_1 = tf.stack([x, x], axis=3)
    reshape_1 = tf.reshape(pack_1, [1, 3, 6, 1])
    pack_2 = tf.stack([reshape_1, reshape_1], axis=2)
    reshape_2 = tf.reshape(pack_2, [1, 6, 6, 1])
    scaled = tf.image.resize(reshape_2, size=(3, 3), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return scaled + x

inp = np.random.standard_normal((1, 3, 3, 1)).astype(np.float32)
save_tflite_model(replicate_by_pack, inp, 'replicate_by_pack')

