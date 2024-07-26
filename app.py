from flask import Flask, render_template, Response
from deepface import DeepFace
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)

face_id_map = {}
current_ids = set()
next_id = 1

def generate_frames():
    global next_id, current_ids, face_id_map
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            try:
                faces = DeepFace.extract_faces(img_path=frame, enforce_detection=False)
                new_ids = set()
                for face in faces:
                    face_embedding = face['embedding']
                    face_found = False
                    for face_id, face_data in face_id_map.items():
                        if DeepFace.verify(face_embedding, face_data['embedding'], model_name='Facenet')['verified']:
                            face_found = True
                            new_ids.add(face_id)
                            break
                    if not face_found:
                        face_id_map[next_id] = {'embedding': face_embedding}
                        new_ids.add(next_id)
                        next_id += 1

                for face_id in current_ids - new_ids:
                    face_id_map.pop(face_id, None)

                current_ids = new_ids

                for face in faces:
                    (x, y, w, h) = face['facial_area']
                    face_id = next_id if next_id in current_ids else next(filter(lambda key: DeepFace.verify(face['embedding'], face_id_map[key]['embedding'], model_name='Facenet')['verified'], face_id_map.keys()))
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f'ID: {face_id}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except:
                continue

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(debug=True)
