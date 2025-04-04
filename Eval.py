from flask import Flask, request, jsonify
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from flask_cors import CORS
from twilio.rest import Client

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend interaction

# Load the trained model
MODEL_PATH = "./cyberbullying_model"
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Twilio Credentials (Replace with your actual credentials)
TWILIO_ACCOUNT_SID = "get ur account sid from twilio"
TWILIO_AUTH_TOKEN = "get ur token from twilio"
TWILIO_PHONE_NUMBER = "+12244903168"
ALERT_PHONE_NUMBER = "add ur number"  # Example: "add ur number"

def predict(text):
    """Predict if a message contains cyberbullying content."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return "Cyberbullying" if predicted_class == 0 else "Not Cyberbullying"

def send_sms_alert(message):
    """Send an SMS alert using Twilio when cyberbullying is detected."""
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=f"⚠️ Cyberbullying detected: {message}",
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )
        print("🚨 SMS Alert Sent!")
    except Exception as e:
        print("❌ Failed to send SMS:", e)

@app.route("/predict", methods=["POST"])
def detect_cyberbullying():
    """API route to detect cyberbullying in messages."""
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    prediction = predict(text)

    # If cyberbullying detected, send SMS alert
    if prediction == "Cyberbullying":
        send_sms_alert(text)

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
