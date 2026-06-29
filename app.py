"""
app.py — Speech Smoothing System v2
Run: python app.py
"""
import os
from flask import Flask
from flask_cors import CORS
from api.routes import api_bp
from config import Config

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    CORS(app, resources={r"/*": {"origins": "*"}})
    app.register_blueprint(api_bp, url_prefix="/api")
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

    @app.route("/health")
    def health():
        return {"status": "ok", "service": "Speech Smoothing System v2"}

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=False, host="0.0.0.0", port=5000)
