import os
import numpy as np
import librosa
import tensorflow as tf
import soundfile as sf
from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, current_user, logout_user, login_required
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
import speech_recognition as sr
import logging
from models import db, User

app = Flask(__name__)
app.config['SECRET_KEY'] = 'shraddha17'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'sunithavidyadhar18@gmail.com'
app.config['MAIL_PASSWORD'] = 'eocb gqzv jdtv wimg'

mail = Mail(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Initialize db with app
db.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

model_path = 'models/history.h5'
model_weights_path = "models/history_weights.h5"

# Ensure the model paths are correct
model = tf.keras.models.load_model(model_path)
model.load_weights(model_weights_path)

# Check model input shape
input_shape = model.input_shape[1:]

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def audio_to_spectrogram(y, sr, n_mels=128):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB

def normalize_spectrogram(S_DB):
    scaler = StandardScaler()
    S_DB_norm = scaler.fit_transform(S_DB.T).T
    return S_DB_norm

def pad_or_truncate(S_DB_norm, max_length=282):
    if S_DB_norm.shape[1] < max_length:
        pad_width = max_length - S_DB_norm.shape[1]
        S_DB_norm = np.pad(S_DB_norm, ((0, 0), (0, pad_width)), mode='constant')
    else:
        S_DB_norm = S_DB_norm[:, :max_length]
    return S_DB_norm

def preprocess_audio(file_path, input_shape, max_length=282):
    try:
        y_audio, sr = load_audio(file_path)
        S_DB = audio_to_spectrogram(y_audio, sr)
        S_DB_norm = normalize_spectrogram(S_DB)
        S_DB_norm = pad_or_truncate(S_DB_norm, max_length)
        S_DB_norm = S_DB_norm[..., np.newaxis]

        # Flatten if required
        if len(input_shape) == 2 and input_shape[1] == S_DB_norm.shape[0] * S_DB_norm.shape[1]:
            S_DB_norm = S_DB_norm.flatten()
        return np.array([S_DB_norm])
    except Exception as e:
        logging.error(f"Error in preprocess_audio: {str(e)}")
        return None

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text.lower()
    except sr.RequestError as e:
        logging.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except sr.UnknownValueError:
        logging.error("Google Speech Recognition could not understand audio")
        return None

@app.route("/", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('microphone'))
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('microphone'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html')

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('microphone'))
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for('microphone'))
    return render_template('signup.html')

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route("/microphone")
@login_required
def microphone():
    return render_template('microphone.html')

@app.route("/predict", methods=['POST'])
@login_required
def predict():
    if 'audio' not in request.files:
        return jsonify({'message': 'No file uploaded'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'message': 'No file selected'}), 400

    filename = secure_filename(audio_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(file_path)

    max_length = 282  # Adjust based on your model's input requirements
    preprocessed_audio = preprocess_audio(file_path, input_shape, max_length=282)

    if preprocessed_audio is None:
        return jsonify({'message': 'Error processing audio file'}), 500

    try:
        # Predict emergency or not using the pre-trained model
        prediction = model.predict(preprocessed_audio)
        class_labels = ['Not Emergency', 'Emergency']
        predicted_class = np.argmax(prediction, axis=1)
        result = class_labels[predicted_class[0]]

        if result == 'Emergency':
            # Transcribe the audio and check for emergency words
            emergency_words = {
                "fire": ["fire", "burning", "smoke"],
                "physical assault": ["attack", "assault", "fight"],
                "dog": ["dog", "bite", "barking"],
                "theft": ["theft", "robbery", "stealing", "thief"]
            }
            transcription = transcribe_audio(file_path)
            detected_emergency = None

            if transcription:
                for emergency_type, words in emergency_words.items():
                    if any(word in transcription for word in words):
                        detected_emergency = emergency_type
                        break

            emergency_message = 'Emergency'
            if detected_emergency:
                emergency_message += f" ({detected_emergency.capitalize()} detected)"

            # Send email to the logged-in user
            msg = Message("Emergency Detected", sender='sunithavidyadhar18@gmail.com', recipients=[current_user.email])
            msg.body = f"An emergency has been detected based on the audio input. {emergency_message}"
            mail.send(msg)

            os.remove(file_path)
            return jsonify({'message': emergency_message, 'emergency': True})
        else:
            os.remove(file_path)
            return jsonify({'message': 'Not Emergency', 'emergency': False})
    except Exception as e:
        logging.error(f"Error in predict: {str(e)}")
        return jsonify({'message': 'Error in prediction'}), 500

if __name__ == "__main__":
    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    with app.app_context():
        if not os.path.exists('site.db'):
            db.create_all()
    app.run(debug=True)
