import sys
import cv2
import numpy as np
import os

# Get the current directory of the script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script's directory
os.chdir(current_directory)

def motoDetected(p):
    if p > 0.6:
        print("Moto detectada" + str(p))

# Now, you can open the file without specifying the full path
LABELS = open("coco.names").read().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
def analyze_frame(frame):
    # Load YOLO model configuration and weights
    config = "yolov7-tiny.cfg"
    #config = "yolov3.cfg"
    #weights = "yolov3.weights"
    weights = "yolov7-tiny.weights"
    net = cv2.dnn.readNetFromDarknet(config, weights)

    # Get frame dimensions
    height, width, _ = frame.shape

    # Preprocess the frame and get detections
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    net.setInput(blob)
    outputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    # Process detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            #scores = detection[:5]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.6:
                box = detection[:4] * np.array([width, height, width, height])
                (x_center, y_center, w, h) = box.astype("int")
                x = int(x_center - (w / 2))
                y = int(y_center - (h / 2))

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply non-maximum suppression to suppress weak, overlapping bounding boxes
    idx = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

    # Draw bounding boxes and labels on the frame
    num_objects_to_process = 3
    if len(idx) > 0:
        for i in idx[:num_objects_to_process]:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = COLORS[classIDs[i]].tolist()
            text = "{}: {:.3f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # if text starts with motorbike call motoDetected()
            if text.startswith("person"):
                motoDetected(confidences[i])

    return frame

# Example usage:
# Read a frame from video capture (cap) using cap.read()
# processed_frame = analyze_frame(frame)
# Display or save processed_frame as needed

# Main function for processing video
# receives video url as parameter
# program should be run as: python myapp2.py http://
def main(video_url):
    # Open video file
    #cap = cv2.VideoCapture("video_source.mp4")
    
    frame_counter = 0
    frame_skip = 1  # Process every 10th frame
    #video_url = "http://192.168.1.39:6677/videofeed?username=&password="
    
    while True:
        # Read frame from the video
        cap = cv2.VideoCapture(video_url)
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1

        # Process every 10th frame
        if frame_counter % frame_skip == 0:
            processed_frame = analyze_frame(frame)

            # Display the frame
            cv2.imshow("Motorbike Detection", processed_frame)
        cap.release()
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close windows
    #cap.release()
    cv2.destroyAllWindows()
video_url = sys.argv[1]
if __name__ == "__main__":
    main(video_url)