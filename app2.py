from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
from ultralytics import YOLO
import supervision as sv

class App:
    def __init__(self):
        self.detection_status = 0
        self.location = [0, 0, 0]
        self.cam = False
        self.cap = None
        self.model = None

    def initialize_stream(self):
        url = "http://192.168.88.2:8081/stream" if self.cam else "http://192.168.88.2:8082/stream"
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
                self.model = YOLO("bestv2.pt")
                print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model due to: {e}")

    def detection(self):
        if self.detection_status == 1 and self.model is None:
            self.load_model()

        while True:
            
            frame_bytes = None
            success, frame = self.read_frame()
            if not success or frame is None:
                continue
            if self.detection_status == 1:
                if self.model is not None:
                    results = self.model.predict(frame, conf=0.1, show=False, verbose=False, iou=0.4)[0]
                    detections = sv.Detections.from_ultralytics(results)
                    self.location = [0, 0, 0]
                    for i, bbox in enumerate(detections.xyxy):
                        bbox_center = (bbox[0] + bbox[2]) / 2
                        frame_third = frame.shape[1] / 3
                        if bbox_center < frame_third:
                            self.location[0] = 1
                        elif bbox_center < frame_third * 2:
                            self.location[1] = 1
                        else:
                            self.location[2] = 1
                    annotated_frame = self.annotate_frame(frame, detections)
                    frame_bytes = self.frame_to_bytes(annotated_frame)
                else:
                    print("Model is not loaded, skipping frame processing.")
                    self.load_model()
            else:
                self.model = None
                frame_bytes = self.frame_to_bytes(frame) if frame is not None else None

            if frame_bytes is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def annotate_frame(self, frame, detections):
        bounding_box_annotator = sv.BoundingBoxAnnotator(color=sv.ColorPalette.from_hex(['#f00000']), thickness=4)
        return bounding_box_annotator.annotate(scene=frame.copy(), detections=detections)

    def frame_to_bytes(self, frame):
        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

app_state = App()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/detection")
def video_detection():
    return StreamingResponse(app_state.detection(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/start-detect")
def start_detect():
    app_state.detection_status = 1
    return {"message": "Detection started"}

@app.get("/stop-detect")
def stop_detect():
    app_state.detection_status = 0
    app_state.model = None
    return {"message": "Detection stopped"}

@app.get("/swap-camera")
def swap_camera():
    app_state.cam = not app_state.cam
    app_state.cap = None
    return {"message": "Camera swapped"}

@app.get("/detectionStatus")
def get_detection_status():
    return {"detectionStatus": app_state.detection_status}

@app.get("/location")
def get_location():
    return {"location": app_state.location}
