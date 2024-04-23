import ultralytics
from ultralytics import YOLO

alpaca = "test_data/alpaca.jpg"
dimssum = "test_data/dimssum.jpg"
giraffe = "test_data/giraffe.jpg"
maneki = "test_data/maneki.jpg"
restaurants = "test_data/restaurants.jpg"
dimssum2 = "test_data/dimsum_screeenshot.jpg"
zoomed_restaurant = "test_data/zoomed_restaurant.jpg"
zoomed_restaurant1 = "test_data/zoomed_restaurant1.jpg"
zoomed_restaurant2 = "test_data/zoomed_restaurant2.jpg"
zoomed_restaurant3 = "test_data/zoomed_restaurant3.jpg"
zoomed_restaurant4 = "test_data/zoomed_restaurant4.jpg"
zoomed_restaurant5 = "test_data/zoomed_restaurant5.jpg"
zoomed_restaurant6 = "test_data/zoomed_restaurant6.jpg"

model = YOLO("models/205trainset_400epoch_model_S.pt")
source_img = [dimssum2,zoomed_restaurant,zoomed_restaurant1,zoomed_restaurant2,zoomed_restaurant3,zoomed_restaurant4,zoomed_restaurant5,zoomed_restaurant6]

#source_img = [alpaca, dimssum, giraffe, maneki, restaurants, dimssum2, zoomed_restaurant,zoomed_restaurant1,zoomed_restaurant2,zoomed_restaurant3,zoomed_restaurant4,zoomed_restaurant5,zoomed_restaurant6]
results = model(source=source_img, conf=0.4, show=False, save=True)