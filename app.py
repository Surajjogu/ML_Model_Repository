from flask import Flask, request, render_template_string
import numpy as np
import pickle
import requests

app = Flask(__name__)

GIT_REPO_URL = "https://raw.githubusercontent.com/Surajjogu/ML_Model_Repository/main/"

def fetch_file(file_name):
    try:
        response = requests.get(GIT_REPO_URL + file_name)
        if response.status_code == 200:
            return response.content
        else:
            return None
    except Exception as e:
        print(f"Error fetching {file_name}: {e}")
        return None

index_html_content = fetch_file("index.html")
model_pkl_content = fetch_file("model.pkl")

if index_html_content is None or model_pkl_content is None:
    print("Failed to fetch necessary files. Exiting...")
    exit(1)

model = pickle.loads(model_pkl_content)

@app.route('/')
def home():
    return render_template_string(index_html_content)

@app.route('/predict', methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    
    output = prediction[0]
    
    return render_template_string(index_html_content, prediction_text='Loan Eligibility = {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
