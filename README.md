# Simple_Gesture_Detection_Prototype
This project implements a deep learning model for gesture detection in a target video. It detects a desired gesture from an image/video taken as input and annotates frames in a target video also provided as an input where the gesture occurs.

# Gesture Detection with Faster R-CNN
This project implements a deep learning model for gesture detection using the Faster R-CNN (Region-based Convolutional Neural Network) architecture. The system takes two inputs: a desired gesture, provided as either an image or a short video clip, and a target video where the gesture is to be detected.

# Objective
The objective of the project is to detect instances of the desired gesture within the target video and annotate the frames where the gesture is detected. By leveraging the capabilities of the Faster R-CNN model, the system aims to accurately identify specific gestures within video data, facilitating tasks such as action recognition and human-computer interaction.

# Features
- Input: Accepts a desired gesture as an image or a short video clip, along with a target video.
- Detection: Utilizes the Faster R-CNN model to detect instances of the desired gesture within the target video.
- Annotation: Annotates the frames of the target video where the gesture is detected with a green square around the gesture and the label "DETECTED" in bright green at the top right corner.
- Output: Provides the annotated video as the output, highlighting the occurrences of the desired gesture within the target video.


# Requirements
- Python 3.x
- OpenCV
- PyTorch
- Torchvision
- NumPy
- Scikit-learn
  

# Credits
This project utilizes the Faster R-CNN model provided by PyTorch's torchvision library for object detection tasks.

