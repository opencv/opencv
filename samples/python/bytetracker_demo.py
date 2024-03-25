#import cv2
import numpy as np
import os
import sys

import site; print(site.getsitepackages())

# Constants
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

FONT_SCALE = 0.7
THICKNESS = 1

# Colors
BLACK = (0, 0, 0)
BLUE = (255, 178, 50)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)

home = os.getenv("HOME")
DETECTIONS_OUTPUT_PATH = os.path.join(home, "files", "det_python.txt")
TRACKINGS_OUTPUT_PATH = os.path.join(home, "files", "tracked_python.txt")
VIDEO_OUTPUT_PATH = os.path.join(home, "files", "output_python.mp4")
COCO_NAMES = os.path.join(home, "files", "coco.names")
#NET_PATH = os.path.join(home, "files", "yolov5s.onnx")
NET_PATH = os.path.join(home, "files", "yolov8s.onnx")
#NET_PATH = os.path.join(home, "files", "yolov8x.onnx")
#NET_PATH = os.path.join(home, "files", "bytetrack_x_mot20.onnx")

PYTHON_PATH = os.path.join(home,"opencv","python","bindings")

sys.path.append(PYTHON_PATH)
import cv2

#print(dir(cv2))
#print(sys.path)

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX

output_codec = cv2.VideoWriter_fourcc(*'XVID')
output_fps = 10
output_size = (768, 576)

class Detection:
    def __init__(self, rect, class_id, confidence):
        self.rect = rect
        self.class_label = class_id
        self.class_score = confidence



def get_color(idx):
    value = idx + 3
    return (int(37 * value % 255), int(17 * value % 255), int(29 * value % 255))

def drawLabel(input_image, label, left, top):
    font_scale = 0.7
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    base_line = 0

    label_size, _ = cv2.getTextSize(label, font_face, font_scale, thickness)
    top = max(top, label_size[1])

    tlc = (left, top)
    brc = (left + label_size[0], top + label_size[1] + base_line)

    cv2.rectangle(input_image, tlc, brc, BLACK, cv2.FILLED)
    cv2.putText(input_image, label, (left, top + label_size[1]), font_face, font_scale, YELLOW, thickness)

def pre_process_image(input_image, net):
    blob = cv2.dnn.blobFromImage(input_image, 1.0 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    return outputs

def post_process_image(input_image, output, class_names, objects):
    class_ids = []
    confidences = []
    boxes = []

    x_factor = input_image.shape[1] / INPUT_WIDTH
    y_factor = input_image.shape[0] / INPUT_HEIGHT
    yolov8 = False

    rows = output[0].shape[1]
    dimensions = output[0].shape[2]

    if dimensions > rows:
        yolov8 = True
        rows = output[0].shape[2]
        dimensions = output[0].shape[1]
        #84x8400 to rows of 1x84 (8400 rows)
        new_output = output[0]
        new_output = new_output.T
        #print(output)
        output = (new_output,)
        #print(output[0].shape)

    #Iterate through 25200 detections.
    data = output[0]
    #print(len(data))
    max_class_score = 0.0

    for i in range(rows):
        #print(len(data[i]))
        if yolov8:
            #print(data[0])
            classes_scores = data[i][4:]
            #print(len(classes_scores))
            class_id = np.argmax(classes_scores)
            max_class_score = classes_scores[class_id][0]
            if max_class_score > SCORE_THRESHOLD:

                confidences.append(max_class_score)
                class_ids.append(class_id)

                x = data[i][0]
                y = data[i][1]
                w = data[i][2]
                h = data[i][3]

                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                boxes.append((left, top, width, height))
        else:
            confidence = data[i][4]
            if confidence >= CONFIDENCE_THRESHOLD:
                classes_scores = data[i][5:]
                class_id = np.argmax(classes_scores)
                max_class_score = classes_scores[class_id]

                if max_class_score > SCORE_THRESHOLD:
                    confidences.append(confidence)
                    class_ids.append(class_id)

                    cx = data[i][0]
                    cy = data[i][1]
                    w = data[i][2]
                    h = data[i][3]

                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    boxes.append((left, top, width, height))

        #data = data[dimensions:]

    #print(confidences)
    nms_indices = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)
    #print(nms_indices)

    #print("max idx",max(nms_indices))
    for i in nms_indices:
        box = boxes[i] #check this part
        #print(box)
        left, top, width, height = box
        class_id = class_ids[i]
        #class_name = class_names[class_id]
        confidence = confidences[i]

        #color = get_color(class_id)
        #cv2.rectangle(input_image, (left, top), (left + width, top + height), color, 2)
        #label = f"{class_name}: {confidence:.2f}"
        #drawLabel(input_image, label, left, top)

        det = Detection((left, top, width, height), class_id, confidence)
        objects.append(det)

    return input_image

def detection_to_mat(objs):
    output = np.zeros((len(objs), 6), dtype=np.float32)

    for i, det in enumerate(objs):
        output[i, 0] = det.rect[0]
        output[i, 1] = det.rect[1]
        output[i, 2] = det.rect[2]
        output[i, 3] = det.rect[3]
        output[i, 4] = det.class_label
        output[i, 5] = det.class_score

    return output

def write_tracks_to_file(tracked_objects, output_path, frame_number):
    with open(output_path, "a") as output_file:
        if output_file.tell() == 0:
            output_file.write("frame, trackId, x, y, width, height, score, classId\n")

        for i in range(tracked_objects.shape[0]):
            x = int(tracked_objects[i, 0])
            y = int(tracked_objects[i, 1])
            width = int(tracked_objects[i, 2])
            height = int(tracked_objects[i, 3])
            class_id = int(tracked_objects[i, 4])
            score = round(tracked_objects[i, 5],3)
            track_id = int(tracked_objects[i, 6])

            output_file.write(f"{frame_number}, {track_id}, {x}, {y}, {width}, {height}, {score}, {class_id}\n")

def write_detections_to_file(objects, output_path, frame_number):
    with open(output_path, "a") as output_file:
        if output_file.tell() == 0:
            output_file.write("frame, trackId, x, y, width, height, score, classId\n")

        for det in objects:
            x = int(det.rect[0])
            y = int(det.rect[1])
            width = int(det.rect[2])
            height = int(det.rect[3])
            class_id = det.class_label
            score = round(det.class_score,3)

            output_file.write(f"{frame_number}, {-1}, {x}, {y}, {width}, {height}, {score}, {class_id}\n")





def main():
    # Load class list
    class_list = []
    with open(COCO_NAMES, 'r') as f:
        class_list = [line.strip() for line in f.readlines()]

    # Load video capture
    capture = cv2.VideoCapture("../data/vtest.avi")
    if not capture.isOpened():
        print("Failed to open the video.")
        return

    # Load tracker
    params = cv2.ByteTracker_Params()
    params.frameRate = int(capture.get(cv2.CAP_PROP_FPS))
    params.frameBuffer = 30
    tracker = cv2.ByteTracker_create(params)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, output_fps, output_size)

    frame_number = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # Load model
        net = cv2.dnn.readNet(NET_PATH)

        # Pre-process image
        detections = pre_process_image(frame, net)

        # Post-process image
        objects = []
        #frame_clone = frame.copy()
        img = post_process_image(frame, detections, class_list, objects)
        objects_mat = detection_to_mat(objects)
        ok, tracked_objects = tracker.update(objects_mat) #I need to hardcode the output

        if ok:
            for i in range(tracked_objects.shape[0]):
                #print(tracked_objects)
                id_ = int(tracked_objects[i, 6])
                color = get_color(id_)
                tlwh_ = (
                    tracked_objects[i, 0],
                    tracked_objects[i, 1],
                    tracked_objects[i, 2],
                    tracked_objects[i, 3]
                )

                cv2.rectangle(img, (int(tlwh_[0]), int(tlwh_[1])),
                            (int(tlwh_[0] + tlwh_[2]), int(tlwh_[1] + tlwh_[3])),
                            color, 2)
                cv2.putText(img, str(id_), (int(tlwh_[0]), int(tlwh_[1]) - 5),
                            FONT_FACE, FONT_SCALE, RED, THICKNESS)

            write_tracks_to_file(tracked_objects, TRACKINGS_OUTPUT_PATH, frame_number)
            write_detections_to_file(objects, DETECTIONS_OUTPUT_PATH, frame_number)

            layers_times = net.getPerfProfile()[0] / cv2.getTickFrequency()
            label = f"Inference time: {layers_times * 1000:.2f} ms"
            cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, RED, THICKNESS)

            cv2.imshow("Output", img)
            writer.write(img)

            if cv2.waitKey(1) == 27:
                break

            frame_number += 1

    writer.release()
    capture.release()
    print("Output video generated")

if __name__ == "__main__":
    main()
