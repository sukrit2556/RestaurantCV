import ultralytics
from ultralytics import YOLO

alpaca = "test_data/alpaca.jpg"
dimssum = "test_data/dimssum.jpg"
giraffe = "test_data/giraffe.jpg"
maneki = "test_data/maneki.jpg"
restaurants = "test_data/restaurants.jpg"
dimssum2 = "test_data/dimsum_screeenshot.jpg"

model = YOLO("best_dimsum.pt")
source_img = [alpaca, dimssum, giraffe, maneki, restaurants, dimssum2]
results = model(source=source_img, conf=0.9, show=False, save=True)