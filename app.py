from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model and vectorizer with error handling
try:
    with open('model.pickle', 'rb') as model_file:
        pac = pickle.load(model_file)
    with open('tranform.pickle', 'rb') as vectorizer_file:
        tfidf_vectorizer = pickle.load(vectorizer_file)
    if not hasattr(tfidf_vectorizer, 'vocabulary_'):
        raise ValueError("The TfidfVectorizer instance is not fitted yet.")
except Exception as e:
    print(f"Error loading model or vectorizer: {str(e)}")
    pac = None
    tfidf_vectorizer = None

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/abstract')
def abstract():
    return render_template('abstract.html')

@app.route('/future')
def future():
    return render_template('future.html')

@app.route('/user')
def user():
    return render_template('user.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/preview', methods=["POST"])
def preview():
    try:
        dataset = request.files['datasetfile']
        if dataset and dataset.filename.endswith('.csv'):
            df = pd.read_csv(dataset, encoding='unicode_escape')
            df.set_index('Id', inplace=True)
            return render_template("preview.html", df_view=df.head())  # Show only the first 5 rows
        else:
            return "Invalid file format. Please upload a CSV file.", 400
    except Exception as e:
        return f"An error occurred while processing the file: {str(e)}", 500

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/chart')
def chart():
    abc = request.args.get('news')
    if not abc:
        return render_template('prediction.html', prediction_text="No input provided.")

    input_data = [abc.strip()]
    try:
        tfidf_test = tfidf_vectorizer.transform(input_data)
        y_pred = pac.predict(tfidf_test)
        label = "Spam" if y_pred[0] == 1 else "No Spam"
    except Exception as e:
        label = f"Error during prediction: {str(e)}"
    
    return render_template('prediction.html', prediction_text=label)

@app.route('/performance')
def performance():
    return render_template('performance.html')

if __name__ == '__main__':
    app.run(debug=True)








