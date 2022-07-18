import os
# from app import app
# import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, Response
from werkzeug.utils import secure_filename
from predict_person import predict_person, get_bbox
from yolo_model import yolo_utils
from predict_emotion import predict_emotion, Emotic, predict_video
import config
import cv2
import torch

app = Flask(__name__)
video = cv2.VideoCapture(0)
app.config["DEBUG"] = True
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
ALLOWED_VIDEO = set(['mp4', 'm4p', 'm4v', 'gif', 'mov'])

device = config.DEVICE
yolo = yolo_utils.prepare_yolo('yolo_model')
yolo = yolo.to(device)
yolo.eval()

model_context = torch.load(os.path.join(config.MODEL_PATH, config.MODEL_CONTEXT))
model_body = torch.load(os.path.join(config.MODEL_PATH, config.MODEL_BODY))
emotic_model = torch.load(os.path.join(config.MODEL_PATH, config.MODEL_EMOTIC))

model_context.to(device)
model_body.to(device)
emotic_model.to(device)

model_context.eval()
model_body.eval()
emotic_model.eval()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_form')
def up_form():
    return render_template('up_image.html')

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_image', methods=['POST'])
def up_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for("up_form"))
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(url_for("up_form"))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded and displayed below')
        bbox = predict_person(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        name_result, emotions, vad = predict_emotion(bbox, os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Predicted Your image')
        return render_template('display_image.html', filename=filename, name_result=name_result)
    else:
        flash('Allowed image types are -> png, jpg, jpeg')
        return redirect(url_for("up_form"))

@app.route('/upload_list_form')
def up_list_form():
    return render_template('up_list_image.html')

@app.route('/upload_list_image', methods=['POST'])
def up_list_images():
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(url_for("up_list_form"))
    files = request.files.getlist('files[]')
    file_names = []
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            bbox = predict_person(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            name_result, emotions, vad = predict_emotion(bbox, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            results.append(name_result)
        else:
            flash('Allowed image types are -> png, jpg, jpeg')
            return redirect(url_for("up_list_form"))

    return render_template('display_list_images.html', filenames=file_names, results=results)

def video_stream_predict(video):
    global device, yolo, model_context, model_body, emotic_model
    while True:
        success, image = video.read()
        image_context = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bbox = get_bbox(yolo, device, image_context)
        try:
            if len(bbox) == 1:
                emotions, vads = predict_video(bbox, image_context, model_context, model_body, emotic_model)
                image = cv2.rectangle(image, (bbox[0][0], bbox[0][1]),(bbox[0][2] , bbox[0][3]), (255, 0, 0), 3)
                cv2.putText(image, vads, (bbox[0][0], bbox[0][1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                for i, emotion in enumerate(emotions):
                    cv2.putText(image, emotion, (bbox[0][0], bbox[0][1] + (i+1)*12), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            else:
                list_emotions, list_vad = predict_video(bbox, image_context, model_context, model_body, emotic_model)
                for i in range(len(bbox)):
                    person = 'person_'+str(i)
                    cv2.rectangle(image, (bbox[i][0], bbox[i][1]),(bbox[i][2] , bbox[i][3]), (255, 0, 0), 3)
                    cv2.putText(image, person, (bbox[i][0], bbox[i][1] - 17), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                    cv2.putText(image, list_vad[i], (bbox[i][0], bbox[i][1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                    for j, emotion in enumerate(list_emotions[i]):
                        cv2.putText(image, emotion, (bbox[i][0], bbox[i][1] + (j+1)*12), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        except:
            continue
        
        ret, jpeg = cv2.imencode('.jpg', image)

        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_stream')
def video_stream():
    global video
    return Response(video_stream_predict(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video_file_form')
def up_video_file_form():
    return render_template('up_video_file.html')

def allowed_video(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO

@app.route('/upload_video', methods=['POST'])
def up_video():
    global device, yolo, model_context, model_body, emotic_model
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for("up_video_file_form"))
    file = request.files['file']
    if file.filename == '':
        flash('No video selected for uploading')
        return redirect(url_for("up_video_file_form"))
    if file and allowed_video(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Video successfully uploaded and displayed below')

        video_stream = cv2.VideoCapture(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        writer = None

        while True:
            (grabbed, frame) = video_stream.read()
            if not grabbed:
                break
            image_context = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                bbox = get_bbox(yolo, device, image_context)
                for pred_idx, pred_bbox in enumerate(bbox):
                    emotions, vads = predict_video(bbox, image_context, model_context, model_body, emotic_model)
                    image_context = cv2.rectangle(image_context, (pred_bbox[0], pred_bbox[1]),(pred_bbox[2] , pred_bbox[3]), (255, 0, 0), 3)
                    cv2.putText(image_context, vads, (pred_bbox[0], pred_bbox[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                    for i, emotion in enumerate(emotions):
                        cv2.putText(image_context, emotion, (pred_bbox[0], pred_bbox[1] + (i+1)*12), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            except Exception:
                pass
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(os.path.join(config.PREDICTED_PATH, 'result_vid.avi'), fourcc, 30, (image_context.shape[1], image_context.shape[0]), True)
            writer.write(cv2.cvtColor(image_context, cv2.COLOR_RGB2BGR))
        writer.release()
        video_stream.release()

        filename = os.path.join('uploads/', filename)
        name_result = os.path.join('predicted/', 'result_vid.avi')
        flash('Predicted Your video')
        return render_template('display_video.html', filename=filename, name_result=name_result)
    else:
        print(file.filename)
        flash('Allowed video types are -> .mp4, .m4p, .m4v, .gif, .mov')
        return redirect(url_for("up_video_file_form"))

app.run()