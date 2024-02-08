from flask import Flask, Response
import cv2  # OpenCV for video processing
from flask_cors import CORS
from ultralytics import YOLO
import supervision as sv
app = Flask(__name__)
CORS(app)
detectionStatus = 0  # Global variable to control the detection status
location = []  # Global variable to store the location
left = 0
middle = 0
right = 0
cam = False
cap =None
model =None

def detection():
    global detectionStatus, location, left, middle, right, cam, cap,model
    model = None
    i = 1
    while True:
        left = 0
        middle = 0
        right = 0
        if cap is None:
            if(cam):
                cap = cv2.VideoCapture("http://192.168.88.2:8081/stream")
            else:
                cap = cv2.VideoCapture("http://192.168.88.2:8082/stream")
            success, frame = cap.read()
        else:
            success, frame = cap.read()
        if detectionStatus == 1:
            if not success:
                break
            if model is None:
                model = YOLO("bestv2.pt")
            results = model.predict(frame, conf=0.1, show=False, verbose=True, iou= 0.4)[0]
            detections = sv.Detections.from_ultralytics(results)
            bounding_box_annotator = sv.BoundingBoxAnnotator(color=sv.ColorPalette.from_hex(['#00ff22']))
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
                location = [left, middle, right]
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            i += 1
            frame = buffer.tobytes()
        else:
            location = [0, 0, 0]
            if not success:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/detection')
def video_detection():
    global detectionStatus
    return Response(detection(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/start-detect')
def start_detect():
    global detectionStatus, model
    detectionStatus = 1
    if(detectionStatus == 1):
        model = YOLO("bestv2.pt")
    return "Detection started"
@app.route('/stop-detect')
def stop_detect():
    global detectionStatus,model
    detectionStatus = 0
    model=None
    return "Detection stopped"
@app.route('/swap-camera')
def swap_camera():
    global cam, cap
    if cam :
        cam = False
    else:
        cam = True
    if(cam):
        cap = cv2.VideoCapture("http://192.168.88.2:8082/stream")
    else:
        cap = cv2.VideoCapture("http://192.168.88.2:8081/stream")
    return "Camera swapped"
@app.route('/detectionStatus')
def get_detection_status():
    global detectionStatus
    return {'detectionStatus': detectionStatus}
@app.route('/location')
def get_location():
    global location
    return {'location': location}
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, threaded=True)