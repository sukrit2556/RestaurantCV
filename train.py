"""import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" """

from ultralytics import YOLO

# Load a model

if __name__ == '__main__':
    # Code that uses multiprocessing
    model = YOLO('models/yolov8m.pt.pt')
    results = model.train(data='myconfig.yaml', epochs=100, imgsz=640)
    model.export()