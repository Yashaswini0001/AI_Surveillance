## Real-Time Emergency Detection

This project records audio in real-time and detects if it is an emergency. It also categorizes the type of emergency, such as fire, dog barking, physical assault, or theft.

## Features
- Real-time audio recording
- Emergency detection
- Categorization of emergencies (fire, dog, physical assault, theft)
- Visual and audio alerts for detected emergencies

## Requirements
- Python 3.8 or higher
- Flask
- TensorFlow
- librosa
- JavaScript and HTML for the frontend

## Setup Instructions

### 1. Clone the Repository
git clone https://github.com/shraddhavp8/real_time_emergency_detection.git
cd real_time_emergency_detection
### 2. Install Dependencies
pip install -r requirements.txt
### 3. Run the Flask Application
flask run
The application will be available at http://127.0.0.1:5000

## Usage
## Recording Audio
Navigate to http://127.0.0.1:5000.
Log in or sign up if you don't have an account.
Use the microphone feature to record audio in real-time.
The audio will be processed, and the result will indicate whether it is an emergency or not, along with the type of emergency.
## File Upload
You can also upload a .wav file from your laptop for analysis.
The application will preprocess the file and classify it as an emergency or non-emergency.
## Alerts
If an emergency is detected, a popup with the emergency type and a related image will appear.
An audio alert will play if the recorded audio contains emergency keywords.
The results will be displayed with red for emergencies and green for non-emergencies.
