from flask import Flask, render_template,request

import model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    try:
        return model.test_input(request.args.get('age'),request.args.get('sex'),request.args.get('cp'),request.args.get('trestbps'),request.args.get('chol'),request.args.get('fbs'),request.args.get('restecg'),request.args.get('thalach'),request.args.get('exang'),request.args.get('oldpeak'),request.args.get('slope'),request.args.get('ca'),request.args.get('thal'))
    except:
        return 'Somerror'
if __name__ == '__main__':
    app.run(debug=False)
