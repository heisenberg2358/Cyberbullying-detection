from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as ai

app = Flask(__name__)

# Proper CORS setup
CORS(app, resources={r"/chat": {"origins": "http://localhost:5173"}})  # Allow only React app

# API Key
API_KEY = 'use gemini api key'
ai.configure(api_key=API_KEY)

# Load Model
models = ai.list_models()
preferred_models = [
    "models/gemini-1.5-pro-latest",
    "models/gemini-1.5-pro-002",
    "models/gemini-1.5-pro",
    "models/gemini-2.0-pro-exp",
    "models/gemini-2.0-flash"
]
model_name = next((m for m in preferred_models if m in [model.name for model in models]), None)
if not model_name:
    raise ValueError("No supported models found.")

model = ai.GenerativeModel(model_name)
chat = model.start_chat()

# Handle OPTIONS preflight request explicitly
@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat_with_bot():
    if request.method == "OPTIONS":
        return _build_cors_prelight_response()  # Handles preflight properly
    
    data = request.get_json()
    message = data.get("message", "")

    if message.lower() == 'bye':
        return _corsify_actual_response(jsonify({"response": "Goodbye!"}))

    try:
        response = chat.send_message(message)
        return _corsify_actual_response(jsonify({"response": response.text}))
    except Exception as e:
        return _corsify_actual_response(jsonify({"response": "Sorry, I am experiencing issues. Please try again later."}))

# Function to handle CORS preflight requests
def _build_cors_prelight_response():
    response = jsonify({"message": "CORS preflight successful"})
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:5173")
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Max-Age", "3600")
    return response

# Function to ensure CORS headers are present in every response
def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:5173")
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
