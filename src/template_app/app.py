"""Flask web application for model deployment (Slide 20).

Provides REST API endpoints to make real-time predictions.
"""

from typing import Dict, Any
from flask import Flask, request, jsonify

from src.utils.logger import get_logger
from src.main.predict import predict, load_trained_model, FEATURE_COLUMNS

logger = get_logger(__name__)


def create_app() -> Flask:
    """Create and configure the Flask web application.

    Returns:
        Flask: Initialized Flask application.
    """
    app = Flask(__name__)

    # Try pre-loading model during startup
    model = None
    try:
        model = load_trained_model()
        logger.info("Trained model pre-loaded successfully for Flask app.")
    except Exception as e:
        logger.warning(f"Model could not be pre-loaded at startup: {e}. It will be loaded on demand.")

    @app.route("/", methods=["GET"])
    def home() -> Dict[str, Any]:
        """Health check and API overview."""
        return jsonify({
            "status": "online",
            "service": "Iris Classification API",
            "endpoints": {
                "health": "/health (GET)",
                "predict": "/predict (POST)"
            },
            "expected_input_format": {
                "features": [
                    [5.1, 3.5, 1.4, 0.2]
                ]
            }
        }), 200

    @app.route("/health", methods=["GET"])
    def health() -> Dict[str, Any]:
        """Check server and model health."""
        return jsonify({
            "status": "healthy",
            "model_loaded": model is not None
        }), 200

    @app.route("/predict", methods=["POST"])
    def handle_predict():
        """Handle prediction requests.

        Accepts JSON payloads in two formats:
        1. Array of samples:
           {"features": [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]]}
        2. Single dictionary:
           {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
        """
        nonlocal model
        if not request.is_json:
            return jsonify({"error": "Request body must be JSON."}), 400

        payload = request.get_json()

        try:
            # Check single object format
            if all(feat in payload for feat in FEATURE_COLUMNS):
                features = [[payload[f] for f in FEATURE_COLUMNS]]
            elif "features" in payload and isinstance(payload["features"], list):
                features = payload["features"]
            else:
                return jsonify({
                    "error": f"Invalid payload format. Must include 'features' list or individual fields: {FEATURE_COLUMNS}"
                }), 400

            # Reload model if it wasn't loaded at start
            if model is None:
                model = load_trained_model()

            predictions = predict(features, model=model)
            return jsonify({
                "status": "success",
                "predictions": predictions
            }), 200

        except Exception as e:
            logger.error(f"Prediction failed with error: {str(e)}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

    return app
