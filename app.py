from flask import Flask, Response
import cv2  # OpenCV for video processing
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

    

    def read_frame(self):
        try:
            if self.cap is None:
                url = "http://127.0.0.1:8082/stream" if self.cam else "http://127.0.0.1:8084/stream"
                self.cap = cv2.VideoCapture(url)
            success, frame = self.cap.read()
            if not success or frame is None:
                raise cv2.error("Stream timeout or overread error")
            return success, frame
        except (cv2.error, urllib.error.URLError) as e:
            print(f"An error occurred while reading from the video stream: {e}")
            if self.cap is not None:
                self.cap.release()  # Close the current VideoCapture
                self.cap = None  # Set to None so a new VideoCapture will be created on the next read_frame call
            return None, None
        
    def detection(self):
        self.model = None
        i = 1
        while True:
            left = 0
            middle = 0
            right = 0
            success, frame = self.read_frame()
            if not success or frame is None:
                continue
            if self.detection_status == 1:
                if not success or frame is None:
                    break
                if self.model is None:
                    self.model = YOLO("bestv2.pt")
                results = self.model.predict(frame, conf=0.1, show=False, verbose=False, iou= 0.4)[0]
                detections = sv.Detections.from_ultralytics(results)
                bounding_box_annotator = sv.BoundingBoxAnnotator(color=sv.ColorPalette.from_hex(['#f00000']), thickness=4)
                annotated_frame = bounding_box_annotator.annotate(
                    scene=frame.copy(),
                    detections=detections
                )
                for i, bbox in enumerate(detections.xyxy):
                    bbox_center = (bbox[0] + bbox[2]) / 2
                    frame_third = annotated_frame.shape[1] / 3
                    if bbox_center < frame_third:
                        left =1
                    elif bbox_center < 2 * frame_third:
                        middle = 1
                    else:
                        right = 1
                    self.location = [left, middle, right]
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                i += 1
                frame = buffer.tobytes()
            else:
                self.location = [0, 0, 0]
                if not success:
                    break
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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
    if app_state.detection_status == 1:
        app_state.model = YOLO("bestv2.pt")
    return "Detection started"

@app.route('/stop-detect')
def stop_detect():
    app_state.detection_status = 0
    app_state.model = None
    return "Detection stopped"

@app.route('/swap-camera')
def swap_camera():
    app_state.cam = not app_state.cam
    app_state.cap = None
    return "Camera swapped"

@app.route('/detectionStatus')
def get_detection_status():
    return {'detectionStatus': app_state.detection_status}

@app.route('/location')
def get_location():
    return {'location': app_state.location}

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)