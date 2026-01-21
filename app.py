from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
import os
import speech_recognition as sr
from pydub import AudioSegment
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from datetime import datetime
import secrets

app = Flask(__name__)
# Generate a secure random secret key
app.secret_key = secrets.token_hex(32)
# Configure session settings
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes
app.config['SESSION_COOKIE_SECURE'] = False  # Set to False for development
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}
CONFIDENCE_THRESHOLD = 0.6

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database setup
def get_db():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            full_name TEXT,
            email TEXT UNIQUE,
            phone TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        db.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            fluency_score REAL,
            repetitions INTEGER,
            prolongations INTEGER,
            blocks INTEGER,
            recording_path TEXT,
            transcription TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        db.commit()

init_db()

# Load model and label encoder
model = load_model('model/trained_model.h5', compile=False)
le = joblib.load('model/label_encoder.pkl')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(file_path):
    try:
        signal, sr = librosa.load(file_path, sr=22050)
        
        if len(signal) < 2048:
            signal = np.pad(signal, (0, 2048 - len(signal)), mode='constant')
        
        n_fft = 2048
        if len(signal) < n_fft:
            n_fft = len(signal) // 2
            n_fft = n_fft if n_fft % 2 == 0 else n_fft - 1

        mfccs = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=40,
            n_fft=n_fft,
            hop_length=512
        )
        
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        features = np.concatenate([mfccs, delta_mfccs, delta2_mfccs])
        
        if features.shape[1] < 120:
            pad_width = 120 - features.shape[1]
            features = np.pad(features, ((0,0), (0, pad_width)), mode='constant')
        else:
            features = features[:, :120]
        
        return features.T
    except Exception as e:
        app.logger.error(f"Error extracting features: {str(e)}")
        raise

@app.route('/')
@app.route('/index.html')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/learn_more.html')
def learn_more():
    return render_template('learn_more.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        phone = request.form.get('phone')

        if not all([username, password, full_name, email]):
            flash('Please fill all required fields', 'error')
            return redirect(url_for('register'))

        try:
            with get_db() as db:
                db.execute('INSERT INTO users (username, password, full_name, email, phone) VALUES (?, ?, ?, ?, ?)',
                          (username, generate_password_hash(password), full_name, email, phone))
                db.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError as e:
            flash('Username or email already exists', 'error')
        except Exception as e:
            app.logger.error(f"Registration error: {str(e)}")
            flash('An error occurred during registration', 'error')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            flash('Please enter both username and password', 'error')
            return redirect(url_for('login'))

        try:
            with get_db() as db:
                user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
            
            if user and check_password_hash(user['password'], password):
                session.clear()  # Clear any existing session data
                session['user_id'] = user['id']
                session['username'] = user['username']
                session.permanent = True  # Make the session persistent
                flash('Login successful!', 'success')
                return redirect(url_for('index'))  # Redirect to index page after login
            else:
                flash('Invalid username or password', 'error')
        except Exception as e:
            app.logger.error(f"Login error: {str(e)}")
            flash('An error occurred during login', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        with get_db() as db:
            user = db.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
            if not user:
                flash('User not found', 'error')
                return redirect(url_for('login'))
            
            sessions = db.execute('''
                SELECT * FROM sessions 
                WHERE user_id = ? 
                ORDER BY session_date DESC
                LIMIT 10
            ''', (session['user_id'],)).fetchall()
            
            # Prepare data for chart
            all_sessions = db.execute('''
                SELECT * FROM sessions 
                WHERE user_id = ? 
                ORDER BY session_date ASC
            ''', (session['user_id'],)).fetchall()
            
            session_dates = [session['session_date'][:10] for session in all_sessions]
            fluency_scores = [session['fluency_score'] * 100 if session['fluency_score'] else 0 for session in all_sessions]
            repetitions = [session['repetitions'] if session['repetitions'] else 0 for session in all_sessions]
        
        return render_template('profile.html', 
                            user=user, 
                            sessions=sessions,
                            session_dates=session_dates,
                            fluency_scores=fluency_scores,
                            repetitions=repetitions)
    except Exception as e:
        app.logger.error(f"Profile error: {str(e)}")
        flash('An error occurred while loading your profile', 'error')
        return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only WAV files are allowed'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        # Classification pipeline
        features = extract_features(file_path)
        features = np.expand_dims(features, axis=0)
        predictions = model.predict(features)[0]
        confidence = np.max(predictions)
        predicted_class = le.inverse_transform([np.argmax(predictions)])[0]
        
        # Transcription pipeline
        recognizer = sr.Recognizer()
        audio = AudioSegment.from_wav(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        converted_path = file_path.replace('.wav', '_conv.wav')
        audio.export(converted_path, format='wav')
        
        try:
            with sr.AudioFile(converted_path) as source:
                audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            transcription = "Could not transcribe audio"
        except sr.RequestError:
            transcription = "Transcription service unavailable"
        finally:
            if os.path.exists(converted_path):
                os.remove(converted_path)

        # Save session data
        with get_db() as db:
            db.execute('''
                INSERT INTO sessions (user_id, fluency_score, repetitions, prolongations, blocks, recording_path, transcription)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                session['user_id'],
                float(confidence),
                predictions[1] * 100 if len(predictions) > 1 else 0,  # Repetitions probability
                predictions[2] * 100 if len(predictions) > 2 else 0,  # Prolongations probability
                predictions[3] * 100 if len(predictions) > 3 else 0,  # Blocks probability
                file_path,
                transcription
            ))
            db.commit()

        # Prepare response
        response = {
            'prediction': predicted_class,
            'confidence': f'{confidence*100:.2f}%',
            'transcription': transcription,
            'probabilities': {cls: f'{prob*100:.2f}%' for cls, prob in zip(le.classes_, predictions)}
        }

        if confidence < CONFIDENCE_THRESHOLD:
            response['message'] = 'Low confidence prediction'

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/predict_live', methods=['POST'])
def predict_live():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio data received'}), 400

    file = request.files['audio']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only WAV files are allowed'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'live_recording.wav')
    file.save(file_path)

    try:
        # Classification pipeline
        features = extract_features(file_path)
        features = np.expand_dims(features, axis=0)
        predictions = model.predict(features)[0]
        confidence = np.max(predictions)
        predicted_class = le.inverse_transform([np.argmax(predictions)])[0]
        
        # Transcription pipeline
        recognizer = sr.Recognizer()
        audio = AudioSegment.from_wav(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        converted_path = file_path.replace('.wav', '_conv.wav')
        audio.export(converted_path, format='wav')
        
        try:
            with sr.AudioFile(converted_path) as source:
                audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            transcription = "Could not transcribe audio"
        except sr.RequestError:
            transcription = "Transcription service unavailable"
        finally:
            if os.path.exists(converted_path):
                os.remove(converted_path)

        # Save session data
        with get_db() as db:
            db.execute('''
                INSERT INTO sessions (user_id, fluency_score, repetitions, prolongations, blocks, recording_path, transcription)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                session['user_id'],
                float(confidence),
                predictions[1] * 100 if len(predictions) > 1 else 0,
                predictions[2] * 100 if len(predictions) > 2 else 0,
                predictions[3] * 100 if len(predictions) > 3 else 0,
                file_path,
                transcription
            ))
            db.commit()

        # Prepare response
        response = {
            'prediction': predicted_class,
            'confidence': f'{confidence*100:.2f}%',
            'transcription': transcription,
            'probabilities': {cls: f'{prob*100:.2f}%' for cls, prob in zip(le.classes_, predictions)}
        }

        if confidence < CONFIDENCE_THRESHOLD:
            response['message'] = 'Low confidence prediction'

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Live prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)