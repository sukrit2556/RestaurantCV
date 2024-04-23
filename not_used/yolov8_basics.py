from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model

# Predict with the model
results = model(source="inference/videos/restaurantLong.mp4", show=True, conf=0.4, save=False, classes=[0,1])  # predict on an image