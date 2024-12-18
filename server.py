import os

from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
from Inference_Faster_R_CNN_Image import train as detect_image
from Inference_Faster_R_CNN_Video import train as detect_video
from train import train as train_main_model
from werkzeug.utils import secure_filename


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['CORS_HEADERS'] = 'Content-Type'
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow WebSocket CORS
UPLOAD_FOLDER = "./"
# Global variable to track training progress and status
training_status = {"status": "Idle", "message": ""}


# Command: Start training
def train_model(params, emit_progress):
    global training_status
    try:
        training_status["status"] = "In Progress"
        training_status["message"] = "Training started"
        emit_progress("Training started...")

        print(params)
        # Simulate training (replace this with actual train logic)
        #train(params)

        training_status["status"] = "Completed"
        training_status["message"] = "Training completed successfully"
        emit_progress("Training completed successfully.")
    except Exception as e:
        training_status["status"] = "Error"
        training_status["message"] = f"Error occurred: {str(e)}"
        emit_progress(f"Error: {str(e)}")


@app.route('/start-training', methods=['POST'])
def start_training():
    global training_status
    data = request.json
    if training_status["status"] == "In Progress":
        return jsonify({"status": "Error", "message": "Training already in progress"}), 400

    thread = threading.Thread(target=train_model, args=(data, lambda msg: socketio.emit('progress', {'message': msg})))
    thread.start()
    return jsonify({"status": "Training started"}), 202

@app.route('/detect-image', methods=['POST'])
def detect_i():
    # Handle image upload
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(file_path)


    conf_threshold = float(request.form.get("conf_threshold", 0.3))
    args_dict = {"file_path": file_path, "conf_threshold": conf_threshold}
    # Perform detection (replace with actual detection logic)
    result_path = detect_image(args_dict)  # Your detection logic here

    return send_file(result_path, mimetype='image/jpeg')

@app.route('/detect-video', methods=['POST'])
def detect_v():
    # Handle image upload
    if 'file' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(file_path)

    conf_threshold = float(request.form.get("conf_threshold", 0.3))
    args_dict = {"file_path": file_path, "conf_threshold": conf_threshold}
    print(args_dict)
    # Perform detection (replace with actual detection logic)
    result_path = detect_video(args_dict)  # Your detection logic here
    print("done detect video")

    return send_file(result_path, mimetype='video/mp4')


# Query: Get training status
@app.route('/training-status', methods=['GET'])
def training_status_api():
    return jsonify(training_status), 200


@socketio.on('connect')
def connect():
    emit('progress', {'message': 'Connected to training server'})


if __name__ == '__main__':
    socketio.run(app, debug=True)
