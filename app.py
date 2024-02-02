from flask import Flask, Response, request
import cv2  # OpenCV for video processing
from flask_cors import CORS
from ultralytics import YOLO
import supervision as sv

app = Flask(__name__)
CORS(app)

detectionStatus = 1  # Global variable to control the detection status

def detection():
    global detectionStatus
    cap = cv2.VideoCapture(0)  # Start capturing the video
    model = YOLO("bestv2.pt")
    i = 1
    while True:
        if detectionStatus == 1:
            success, frame = cap.read()
            if not success:
                break
            results = model.predict(frame, conf=0.1, show=False, line_width=2, verbose=False, iou= 0.01)[0]
            detections = sv.Detections.from_ultralytics(results)
            bounding_box_annotator = sv.BoundingBoxAnnotator(color=sv.ColorPalette.from_hex(['#00ff22']))
            annotated_frame = bounding_box_annotator.annotate(
                scene=frame.copy(),
                detections=detections
            )
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            print(i,"run")
            i += 1
            annotated_frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + annotated_frame + b'\r\n')
        else:
            success, frame = cap.read()
            if not success:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/detection')
def video_detection():
    global detectionStatus
    detectionStatus = request.args.get('detectionStatus', default = 1, type = int)
    return Response(detection(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# create one api endpoint to get detectection status
@app.route('/detectionStatus')
def get_detection_status():
    global detectionStatus
    return {'detectionStatus': detectionStatus}


if __name__ == '__main__':
    app.run(debug=True)