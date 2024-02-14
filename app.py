from flask import Flask, Response
import cv2
from flask_cors import CORS
from ultralytics import YOLO
import supervision as sv
import urllib.error

class App:
    def __init__(self):
        self.detection_status = 0
        self.location = [0, 0, 0]
        self.cam = False
        self.cap = None
        self.model = None
    def initialize_stream(self):
        url = 0 if self.cam else 2
        self.cap = cv2.VideoCapture(url)

    def read_frame(self):
        if self.cap is None:
            self.initialize_stream()
        success, frame = self.cap.read()
        if not success or frame is None:
            print("Stream timeout or overread error")
            self.cap.release()
            self.cap = None
            return None, None
        return success, frame
    def load_model(self):
        try:
            if self.model is None:
                print("Attempting to load model...")
                self.model = YOLO("/home/a2tech/Desktop/pojek-rhino/rhino-backend/bestv2.pt")  # Consider using an absolute path
                print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model due to: {e}")


    def detection(self):
        if self.detection_status == 1 and self.model is None:
            self.load_model()

        while True:
            frame_bytes = None  # Initialize frame_bytes at the start of the loop
            success, frame = self.read_frame()
            if not success or frame is None:
                continue
            if self.detection_status == 1:
                
                if self.model is not None:
                    results = self.model.predict(frame, conf=0.1, show=False, verbose=False, iou=0.4)[0]
                    detections = sv.Detections.from_ultralytics(results)
                    annotated_frame = self.annotate_frame(frame, detections)
                    frame_bytes = self.frame_to_bytes(annotated_frame)
                else:
                    print("Model is not loaded, skipping frame processing.")
                    self.load_model()
                    # Optionally, handle the case where frame processing is skipped
            else:
                self.model = None 
                # Ensure there's a branch that assigns a value to frame_bytes even if detection_status is not 1
                # For example, convert the original frame to bytes if no detection is performed
                frame_bytes = self.frame_to_bytes(frame) if frame is not None else None

            # Only yield frame_bytes if it's not None
            if frame_bytes is not None:
                yield self.format_frame(frame_bytes)


    def annotate_frame(self, frame, detections):
        bounding_box_annotator = sv.BoundingBoxAnnotator(color=sv.ColorPalette.from_hex(['#f00000']), thickness=4)
        return bounding_box_annotator.annotate(scene=frame.copy(), detections=detections)

    def frame_to_bytes(self, frame):
        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

    def format_frame(self, frame_bytes):
        return (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

app = Flask(__name__)
CORS(app)
app_state = App()


@app.route('/detection')
def video_detection():
    return Response(app_state.detection(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start-detect')
def start_detect():
    app_state.detection_status = 1
    return "Detection started"

@app.route('/stop-detect')
def stop_detect():
    app_state.detection_status = 0
    app_state.model = None  # Ensuring model is cleared when detection stops
    return "Detection stopped"

@app.route('/swap-camera')
def swap_camera():
    app_state.cam = not app_state.cam
    app_state.cap = None  # Force to refresh the video capture in the next frame read
    return "Camera swapped"

@app.route('/detectionStatus')
def get_detection_status():
    return {'detectionStatus': app_state.detection_status}

@app.route('/location')
def get_location():
    return {'location': app_state.location}

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=9099)