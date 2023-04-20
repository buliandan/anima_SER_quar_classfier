# -*- coding: utf-8 -*-
import os
# from flask import Flask,render_template,request
from flask import Flask, request, url_for, send_from_directory,render_template
from werkzeug.utils  import secure_filename
from predict_emo import predict
from predict_emo import mp3_to_wav

'''play with web '''

# 设置哪些媒体格式的文件可以上传
ALLOWED_EXTENSIONS = set(['mp3', 'wav'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getcwd() + '/upload_files'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

 
@app.route('/in', methods=['GET', 'POST'])
# @app.route('/in')
def uploaded_file():
    print('filename:')
    infer_res="模型的输出结果往这里放"
    if request.method == 'GET':
        infer_res="模型的输出结果往这里放"
    return render_template('web.html', hinfer_res=infer_res)

    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
 
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            #mp3转为wav
            if filename[-1]=='3':
                in_filepath = os.path.join('upload_files', filename)
                filename = filename.strip('.mp3') + '.wav'
                out_filepath = os.path.join('upload_files', filename)
                mp3_to_wav(in_filepath, out_filepath)
                print("已转换" + in_filepath + "至" + out_filepath + "，并且保存")

            file_url = url_for('uploaded_file', filename=filename)
            print('filename:',filename)
            # 模型推理
            # res=''
            res=predict('upload_files/'+filename)
            # 模型结果输出
            infer_res='预测结果:'+str(res)
            return render_template('web.html', hinfer_res=infer_res)

    return render_template('web.html')
 
 
if __name__ == '__main__':
    app.run()
