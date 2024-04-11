# Load YOLOv8n-cls, train it on mnist160 for 3 epochs and predict an image with it

from ultralytics import YOLO

model = YOLO('weights/yolov8n-cls.pt')  # load a pretrained YOLOv8n classification model
model.train(data='datasets/animals', epochs=1)  # train the model
print("prediction:")
model('inference/images/cat.jpeg')  # predict on an image
