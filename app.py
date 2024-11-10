from flask import Flask, render_template, request, jsonify
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
model = load_model('model_files/medium_model.h5')
with open('model_files/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Define the maximum sequence length (you should set this to your original max_sequence_len)
max_sequence_len = 50  # Adjust as per your model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_next_word', methods=['POST'])
def predict_next_word():
    data = request.get_json()
    seed_text = data.get('seed_text')

    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)
    
    # Get the top 5 probable words
    top_5_indices = predicted_probs[0].argsort()[-5:][::-1]
    suggestions = []

    for idx in top_5_indices:
        for word, index in tokenizer.word_index.items():
            if index == idx:
                suggestions.append(word)
                break

    return jsonify({'suggestions': suggestions})

if __name__ == '__main__':
    app.run(debug=True)
