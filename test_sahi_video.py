from sahi.predict import get_prediction
from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
import cv2
import time

# Assuming you've already set up your model as shown in the docs
model_path = "weights/yolov8xl.pt"
download_yolov8s_model(model_path)
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=model_path,
    confidence_threshold=0.3,
    device="cuda:0",  # or 'cuda:0' for GPU

)

# Function to process and visualize incoming images on-the-fly
def process_and_visualize(image):
    result = get_prediction(image, detection_model)
    # Access the object prediction list
    print(result.object_prediction_list)
    result.export_visuals(export_dir="output_sahi")

    # Convert to COCO annotation, COCO prediction, imantics, and fiftyone formats
    for item in result.to_coco_annotations()[:]:
        if item['category_id'] == 0:
            print("type = ", item['category_name'], end=" | ")
            print("bbox = ", item['bbox'], end=" | ")
            print("conf = ", item['score'])
            x_min, y_min, width, height = item['bbox']
            top_left = (int(x_min), int(y_min))
            bottom_right = (int(x_min + width), int(y_min + height))
            # Set color for the rectangle (BGR format)
            color = (0, 255, 0)  # Green for better visibility

            # Draw the rectangle on the image
            cv2.rectangle(image, top_left, bottom_right, color, 2)  # Adjust thickness as needed

    # Display the image with the plotted bounding box
    cv2.imshow("Image with COCO Annotation", image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
    # Display or further process the visualized results as needed

image = cv2.imread("output_frame1.jpg")
start = time.time()
process_and_visualize(image)
end = time.time()
elapsed_time = end - start

print(f"Elapsed time: {elapsed_time:.5f} seconds")