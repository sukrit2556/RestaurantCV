import cv2

def reverse_video(input_video_path, output_video_path):
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a VideoWriter object to write the reversed video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # You can use other codecs as well
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Read frames and write them in reverse order
    for frame_number in range(total_frames - 1, -1, -1):
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()

        # Write the frame to the output video
        if ret:
            out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage:
input_video_path = './inference/videos/test_occupied.mp4'
output_video_path = 'result.avi'
reverse_video(input_video_path, output_video_path)
