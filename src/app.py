
from flask import Flask, make_response, request,render_template
import io
from io import StringIO
import csv
import pandas as pd
import numpy as np
import pickle
import os

filename =r'E:\Nidhi\MLOps training\MLOps_Code\Practice_ML\Iris\saved_models\model_rfc2.pickle'
#os.chdir(r'E:\Nidhi\MLOps training\MLOps_Code\Practice_ML\Iris')

app= Flask(__name__, template_folder=r'E:\Nidhi\MLOps training\MLOps_Code\Practice_ML\Iris\template')
def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


@app.route('/')
def main():
    return render_template('home.html')

@app.route('/transform', methods=["POST"])
def transform_view():
    f = request.files['data_file']
    if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    for row in csv_input:
        print(row)

    stream.seek(0)
    result = transform(stream.read())
    df = pd.read_csv(StringIO(result))
    df = df.drop(['Id'],axis=1)

    # load the model from disk
    load_model = pickle.load(open(filename, 'rb'))
    df['prediction'] = load_model.predict(df)

    

    return render_template('simple.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)


if __name__ =='__main__':
    from werkzeug.serving import run_simple
    app.run(debug=True)

  