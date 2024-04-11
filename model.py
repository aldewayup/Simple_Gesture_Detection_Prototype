import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from sklearn.metrics import classification_report

model = fasterrcnn_resnet50_fpn(pretrained=False, progress=True)
model.load_state_dict(torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth", progress=True))

model.eval()

def preprocess_image(image):
    if image is None:
        raise ValueError("Image is None")
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def detect_gesture(frame,model,input_tensor):
    with torch.no_grad():
        prediction = model(input_tensor)

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']


    desired_gesture_label = 1
    gesture_detection = desired_gesture_label in labels

    return gesture_detection, boxes

def annotate_frame(frame, gesture_detection, boxes):
    if gesture_detection:
        for box in boxes:
            cv2.rectangle(frame,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),2)

        cv2.putText(frame,"DETECTED",(frame.shape[1]-200,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    return frame

desired_gesture_path =  r'C:\Users\sai44\OneDrive\Desktop\QuazaAI\gestureinput.mov'

target_video_path = r'C:\Users\sai44\OneDrive\Desktop\QuazaAI\targetvideo.mp4'

if desired_gesture_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
    desired_gesture = cv2.imread(desired_gesture_path)
    input_data = preprocess_image(desired_gesture)
else:
    frames = preprocess_video(desired_gesture_path)

cap = cv2.VideoCapture(target_video_path)
output_path = r'C:\Users\sai44\OneDrive\Desktop\QuazaAI\output_video.mp4'

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))

ground_truth = []
predictions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    input_tensor = preprocess_image(frame)

    gesture_detected, boxes = detect_gesture(frame,model,input_tensor)

    annotated_frame = annotate_frame(frame, gesture_detected, boxes)
    ground_truth.append(1 if gesture_detected else 0)
    predictions.append(1 if gesture_detected else 0)
    out.write(annotated_frame)

    #cv2.imshow('Gesture Detection', annotated_frame)

    if cv2.waitKey(1)  & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(classification_report(ground_truth, predictions))