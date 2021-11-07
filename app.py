from flask import Flask, render_template, request, jsonify
import os

import utilities.predict as predict

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpeg', 'mp4', 'avi'}
VIDEO_FORMAT = {'mp4', 'avi'}
CLASS_NAMES = ['god nilos', 'khedive ismail', 'king fouad I', 'king thutmose III', 'mohamed ali', 'pen-hery the surveyor', 'pen menkh the governer of dendara', 'sphinx of king amenemhat III', 'the protective goddesses', 'writer']
#{'god_nilos': 0, 'khedive_ismail': 1, 'king_fouad_I': 2, 'king_thutmose_III': 3, 'mohamed_ali': 4, 'pen_hery_the_surveyor': 5, 'pen_menkh_the_governer_of_dendara': 6, 'sphinx_of_king_amenemhat_III': 7, 'the_protective_goddesses': 8, 'writer': 9}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'

## Start functions

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

## End functions

## Start home

@app.route("/", methods = ['GET', 'POST'])
def home():
    if(request.method == 'POST'):
        images = []
        videos = []
        notAllowedFiles = []
        emptyFiles = []
        data_list = []
        # check if the post request has the file part
        if 'files[]' not in request.files:
            return "error"

        files = request.files.getlist('files[]')
        
        for file in files:
            if file and allowed_file(file.filename):
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                if os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], file.filename)) == 0:
                    emptyFiles.append(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                else:
                    if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in VIDEO_FORMAT:
                        videos.append(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                    else:
                        images.append(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            else:
                notAllowedFiles.append(file.filename)

        if(len(images) != 0):
            indexs = predict.recognise_image(images)
            predictions = [CLASS_NAMES[index] for index in indexs]
            data_list += list(zip(images, predictions))
            print([f'prediction is {CLASS_NAMES[index]}' for index in indexs])
            print(data_list)

        if(len(videos) != 0):
            prediction = predict.recognise_video(videos)
            predictions = [prediction]
            data_list += list(zip(videos, predictions))
            print(data_list)

        if(len(notAllowedFiles) != 0 or len(emptyFiles) != 0):
            return jsonify(status = False, msg = "Your Files are uploded", emptyFiles = emptyFiles, notAllowedFiles = notAllowedFiles, uploadedFiles = images)
        else:   
            return jsonify(status = True, msg = "Your Files are uploded", emptyFiles = emptyFiles, notAllowedFiles = notAllowedFiles, result = data_list)

    return render_template("upload.html", pagename = "Home Page")

## End home

## Start API

@app.route("/api", methods = ['GET', 'POST'])
def api():
    response = {}
    response['prediction'] = 'No prediction'
    if(request.method == 'POST'):
        mobile_request = request.files.get('image')
        if(mobile_request == None):
            mobile_request = request.files.get('video')
            mobile_request.save(os.path.join(app.config['UPLOAD_FOLDER'], mobile_request.filename))
            prediction = predict.recognise_video(os.path.join(app.config['UPLOAD_FOLDER'], mobile_request.filename))
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], mobile_request.filename))
            #response = {'prediction' : CLASS_NAMES[index] for index in prediction}
            response['prediction'] = CLASS_NAMES[prediction]
        mobile_request.save(os.path.join(app.config['UPLOAD_FOLDER'], mobile_request.filename))
        prediction = predict.recognise_image_api(os.path.join(app.config['UPLOAD_FOLDER'], mobile_request.filename))
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], mobile_request.filename))
        #response = {'prediction' : CLASS_NAMES[index] for index in prediction}
        response['prediction'] = CLASS_NAMES[prediction]
        print(response)
        return response
    else:
        return response

## End API

if __name__ == '__main__':
    app.run(debug = True)