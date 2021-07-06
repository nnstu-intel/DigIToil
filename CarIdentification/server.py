import json
import logging
import os
import time
from flask import Flask, jsonify, request, render_template
from werkzeug import secure_filename

from ISU_vision import car

app = Flask(__name__)
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
c = car()


def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath


@app.route('/', methods=['GET', 'POST'])
def api_root():
    app.logger.info(PROJECT_HOME)
    if request.method == 'POST':
        app.logger.info(app.config['UPLOAD_FOLDER'])
        print(request.files)
        img = request.files['file']
        print(img)
        img_name = secure_filename(img.filename)
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        app.logger.info("saving {}".format(saved_path))
        img.save(saved_path)
        time.sleep(1)
        
        return jsonify(c.info(saved_path))
    else:
        return render_template('test.html')


if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=False, port=80)
