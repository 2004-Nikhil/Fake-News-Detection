from flask import Flask, render_template, request
import pickle
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from tensorflow import keras
import re
from nltk.corpus import stopwords
from flask import jsonify
stop_words = set(stopwords.words('english'))

# Load the trained classification models
modelLog = pickle.load(open('models/logistic.sav', 'rb'))
modelsvm = pickle.load(open('models/svm.sav', 'rb'))
modelrf = pickle.load(open('models/randomforest.sav', 'rb'))
modelbag = pickle.load(open('models/bagging.sav', 'rb'))
modelsoft = pickle.load(open('models/voting.sav', 'rb'))
modelbayes = pickle.load(open('models/bayes.sav', 'rb'))
modelLSTM = keras.models.load_model('models/LSTM.h5')
modelCNN = keras.models.load_model('models/CNN.h5')
modelAttention = keras.models.load_model('models/Attention.h5')
modelBiLSTM = keras.models.load_model('models/BiLSTM.h5')
modelBiLSTM_CNN = keras.models.load_model('models/BiLSTMCNN.h5')
modelGru = keras.models.load_model('models/GRU.h5')
modelBiLSTMCNNa = keras.models.load_model('models/BiCNNa.h5')
modelBiA = keras.models.load_model('models/BiA.h5')


# Load the trained Doc2Vec model
infermod=Doc2Vec.load("models/doc2vec.model")

def preprocess_text(text):
    # remove the text with garbage values like emojis
    text = text.encode('ascii', 'ignore').decode('ascii')

    # remove the text with html tags
    text = re.sub(r'<.*?>', '', text)

    # remove the text with url or links like http or www
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)

    # convert to lower case
    text = text.lower()

    # remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # remove special characters
    text = re.sub(r'\W+', ' ', text)

    text = re.sub(r'\s+', ' ', text)
    return text


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['text']
        model_type = request.form['model']
        message = preprocess_text(message)
        words = message.split()
        # Convert the words into infer vector
        infer_vector = infermod.infer_vector(words)
        vectors = np.array(infer_vector).reshape(1, -1)
        
        # check which model is selected
        if model_type == "Logistic Regression":
            ans = modelLog.predict(vectors)[0]
        elif model_type == "SVM":
            ans = modelsvm.predict(vectors)[0]
        elif model_type == "Random Forest":
            ans = modelrf.predict(vectors)[0]
        elif model_type == "Bagging":
            ans = modelbag.predict(vectors)[0]
        elif model_type == "Soft Voting":
            ans = modelsoft.predict(vectors)[0]
        elif model_type == "Bayes":
            ans = modelbayes.predict(vectors)[0]
        elif model_type == "LSTM":
            ans = modelLSTM.predict(vectors)[0]
        elif model_type == "CNN":
            ans = modelCNN.predict(vectors)[0]
        elif model_type == "Attention":
            ans = modelAttention.predict([vectors,vectors])[0]
        elif model_type == "BiLSTM":
            ans = modelBiLSTM.predict(vectors)[0]
        elif model_type == "BiLSTM CNN":
            ans = modelBiLSTM_CNN.predict(vectors)[0]
        elif model_type == "GRU":
            ans = modelGru.predict(vectors)[0]
        elif model_type == "BiLSTM CNN Attention":
            ans = modelBiLSTMCNNa.predict(vectors)[0]
        elif model_type == "BiLSTM Attention":
            ans = modelBiA.predict(vectors)[0]
            
        print(ans)
        if model_type == "Logistic Regression" or model_type == "SVM" or model_type == "Random Forest" or model_type == "Bagging" or model_type == "Soft Voting" or model_type == "Bayes":
            pass
        else:
            ans=float(ans)
            if ans>0.5:
                ans=1
            else:
                ans=0
        ans=int(ans)
        print(ans)
    label = 'Real' if ans == 1 else 'Fake'
    return jsonify({'prediction': ans, 'label': label})

if __name__ == '__main__':
    app.run(debug=True)