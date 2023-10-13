import os
import os.path as osp
import argparse
import cv2
import numpy as np
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument(
    "testimg_path",type=str, 
    help="the path of the image you want to adding bounding box"
)
CLASSES = yaml_load(check_yaml('coco128.yaml'))['names']

COLOR_MAP = [
    [113, 88, 194], [31, 102, 175], [11, 144, 50], 
    [141, 86, 22], [102, 241, 115], [228, 113, 146], 
    [180, 30, 22], [143, 97, 12], [186, 60, 46], [7, 46, 82], 
    [151, 254, 142], [57, 80, 213], [56, 61, 251], 
    [161, 120, 101], [20, 5, 65], [60, 102, 117], 
    [219, 199, 218], [53, 213, 75], [1, 12, 111], 
    [85, 82, 88], [140, 242, 149], [74, 71, 3], 
    [12, 99, 105], [114, 51, 191], [196, 177, 240], 
    [8, 38, 222], [45, 128, 142], [151, 70, 195], 
    [57, 131, 15], [240, 79, 23], [1, 230, 181], 
    [28, 240, 202], [93, 111, 75], [145, 105, 61], 
    [106, 64, 6], [83, 134, 34], [206, 186, 185], 
    [243, 153, 23], [23, 213, 140], [94, 144, 58], 
    [217, 216, 229], [216, 202, 172], [151, 175, 132], 
    [178, 101, 15], [57, 103, 139], [68, 118, 241], 
    [249, 55, 244], [88, 63, 246], [106, 50, 177], 
    [186, 253, 182], [69, 155, 170], [50, 155, 82], 
    [128, 116, 235], [123, 83, 221], [241, 66, 44], 
    [114, 68, 236], [116, 106, 36], [198, 152, 222], 
    [64, 68, 122], [207, 34, 215], [83, 199, 140], 
    [189, 214, 111], [214, 230, 107], [81, 248, 106], 
    [37, 94, 186], [116, 15, 188], [66, 180, 87], 
    [171, 200, 14], [205, 103, 83], [221, 92, 177], 
    [93, 137, 220], [18, 79, 251], [88, 243, 155], 
    [136, 208, 232], [121, 115, 245], [12, 213, 244], 
    [174, 155, 25], [42, 47, 82], [56, 28, 90], [103, 33, 105]
]


class Opencv_Yolov8:

    def __init__(self, yolov8_onnx_path:os.PathLike, min_confidence:float=0.25) -> None:
        self.model:cv2.dnn.Net = cv2.dnn.readNetFromONNX(yolov8_onnx_path)
        self.target_size = (640,640)
        self.min_confidence = min_confidence
        self.img_size_threshhold=1e6
        self.fontstyle= cv2.FONT_HERSHEY_SIMPLEX

    def preprocess(self, img:np.ndarray)->np.ndarray:
        float_img = img.astype(np.float32)
        length = max(img.shape)
        input_img = np.zeros((length, length, 3), np.float32)
        input_img[0:img.shape[0], 0:img.shape[1]] = float_img
        #as square image with zero padding 
        input_blob = cv2.dnn.blobFromImage(
            image=input_img, scalefactor=(1/255.0),
            size=self.target_size ,swapRB=True
        )
        return input_blob
       
    def __call__(self, img:np.ndarray)->np.ndarray:

        boxes, scores, class_ids, result_boxes =\
            self.__predict(out=self.forward(self.preprocess(img=img)))
        
        img_out = img.copy()
        scale = max(np.array(img.shape[:2])/ 640.0)
        
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            self.draw_bounding_box(
                img_out, class_ids[index], scores[index], 
                round(box[0] * scale), round(box[1] * scale),
                round((box[0] + box[2])*scale), 
                round((box[1] + box[3])*scale)
            )
        return img_out
 
    def forward(self, input_img:np.ndarray)->np.ndarray:
        self.model.setInput(input_img)
        out = (self.model.forward())[0]
        out = np.array([cv2.transpose(out)])
        return out
    
    def __predict(self,out:np.ndarray)->tuple:
    
        boxes, scores, class_ids = [], [], []

        for i in range(out.shape[1]):
            bbox, classes_scores = out[0][i][:4], out[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = \
                    cv2.minMaxLoc(classes_scores)
            
            if maxScore > self.min_confidence: 
                # box : [left x , up y, width,  height]
                box = [
                    bbox[0] - (bbox[2]/2), bbox[1] - (bbox[3]/2),
                    bbox[2], bbox[3] 
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)
            
        return  boxes, scores, class_ids ,\
            cv2.dnn.NMSBoxes(boxes, scores, self.min_confidence, 0.45, 0.5)
    
    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = f'{CLASSES[class_id]}({confidence:.2f})'
        color = COLOR_MAP[class_id]
        tsize , fontthick = 0.7, 1
        if img.shape[0]*img.shape[1] > self.img_size_threshhold :
            tsize,fontthick = 2.3, 2
        boxthick = int((max(img.shape[0], img.shape[1]))/200)
        textpos =((x+x_plus_w)//2, (y+y_plus_h)//2)
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, boxthick)

        def add_text():
            ts , _ = cv2.getTextSize(label, self.fontstyle, fontScale=tsize, thickness=fontthick)
            sub_img = img[textpos[1]-ts[1]:textpos[1], textpos[0]:textpos[0]+ts[0]]
            black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
            res = cv2.addWeighted(sub_img, 0.5, black_rect, 0.5, 1.0)
            img[textpos[1]-ts[1]:textpos[1], textpos[0]:textpos[0]+ts[0]] =res
            cv2.putText(img, label, textpos, self.fontstyle, tsize, (255,255,255), fontthick)
        
        add_text()
    

def download_and_export_model(yolov8_dir:os.PathLike)->os.PathLike:
    
    modelpath = osp.join(yolov8_dir, "yolov8net.onnx")
    print(f"model path : {modelpath}")
    if os.path.exists(modelpath):
        print("find yolov8 onnx model")
        return modelpath

    if not osp.exists(yolov8_dir):
        os.mkdir(yolov8_dir)
    
    print("download model ..")
    original_model = YOLO("yolov8n.pt")
    original_model.export(format="onnx",opset=12)
    os.replace('yolov8n.onnx',modelpath)
    os.replace('yolov8n.pt', osp.join(yolov8_dir, 'yolov8n.pt'))
    return modelpath

def main(testimg_path:os.PathLike):
    print(testimg_path)
    yolov8_onnx_path = download_and_export_model(
        yolov8_dir=osp.join("model")
    )
    cvyolo = Opencv_Yolov8(yolov8_onnx_path=yolov8_onnx_path)
    testimg = cv2.imread(testimg_path)
    prediction = cvyolo(img=testimg)
    folder, imgname = osp.split(testimg_path)
    outname=osp.join(folder,f"bbox_{imgname}")
    print(outname, end=",write ")
    print(cv2.imwrite(outname,prediction))
    print("="*40)

if __name__ == "__main__":
    args = parser.parse_args()
    main(testimg_path=args.testimg_path)