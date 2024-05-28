import cv2
import numpy as np
import time
import requests
from time import sleep


# Load Yolo
net = cv2.dnn.readNet("yolov.weights", "yolov.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def trigger():
    url = "https://www.fast2sms.com/dev/bulkV2"
    querystring = {"Enter you Fast sms API","message":"Enter your message","language":"english","route":"q","numbers":"Enter you number"}
    headers = {
        'cache-control': "no-cache"
        }
    response = requests.request("GET", url, headers=headers, params=querystring)
    print(response.text)


# Loading image
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

while True:
    _, frame = cap.read()
    frame_id == 1
    height, width, channels = frame.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320 ), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                print(indexes)
                    
                    
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        color = colors[class_ids[i]]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)
                print(label)

                if label=='aeroplane':
                    print("Alert Message")
                    trigger()
                    sleep(5)

    
                if label=='cell phone':
                    print("Mobile detected")
                    trigger()
                    sleep(20)
                
                elapsed_time =time.time() - starting_time
                fps = frame_id / elapsed_time
                cv2.putText(frame, "FPS: " + str(fps), (10, 30), font, 3, (0, 0, 0), 1)
                cv2.imshow("Image", frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break
                        

cv2.destroyAllWindows()
