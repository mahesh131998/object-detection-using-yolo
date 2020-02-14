import os
import cv2
import numpy as np
import datetime
import time

weights = "yolov3.weights" 
yolo_cfg = "yolov3.cfg" 
name = str(datetime.datetime.now().strftime('%d-%b-%Y %H-%M')) 
cam = cv2.VideoCapture(0) 
time.sleep(2)
ret, frame = cam.read()
img_name = "{}.jpg".format(name) 
cv2.imwrite(img_name, frame)
print("{} written!".format(img_name))
cam.release() 
j = 0
def get_output_layers(net): 
 layer_names = net.getLayerNames()
 output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
 return output_layers
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h): 
 label = str(classes[class_id])
 color = COLORS[class_id]
 cv2.rectangle(img, (int(x),int(y)), (int(x_plus_w),int(y_plus_h)), color, 2) 
 cv2.putText(img, label, (int(x)-10,int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

image = cv2.imread(img_name)
Width = image.shape[1] 
Height = image.shape[0]
scale = 0.00392 
classes = None 
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, 'yolov3.txt')
with open(my_file, 'r') as f:
 classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3)) 
net = cv2.dnn.readNet(weights, yolo_cfg) 
blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False) 
net.setInput(blob)
outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4
for out in outs:
 for detection in out:
     scores = detection[5:]
     class_id = np.argmax(scores) 
     confidence = scores[class_id]
     if confidence > 0.3: 
         center_x = int(detection[0] * Width)
         center_y = int(detection[1] * Height)
         w = int(detection[2] * Width)
         h = int(detection[3] * Height)
         x = center_x - w / 2
         y = center_y - h / 2
         class_ids.append(class_id)
         confidences.append(float(confidence))
         boxes.append([x, y, w, h]) 
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
for i in indices:
 i = i[0]
 box = boxes[i]
 x = box[0]
 y = box[1]
 w = box[2]
 h = box[3]
 draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w),round(y+h))
 count = str(len(indices)) 
 cv2.imshow("object detected " + count, image) 
 cv2.imwrite("object-detected " + img_name, image) 
 cv2.waitKey() 
 cv2.destroyAllWindows() 