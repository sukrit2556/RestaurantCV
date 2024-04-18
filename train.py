from ultralytics import YOLO

# Load a model

if __name__ == '__main__':
    # Code that uses multiprocessing
    model = YOLO('yolov8x.pt')
    results = model.train(data='myconfig.yaml', epochs=30, imgsz=640)
    model.export()