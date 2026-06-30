"""
api.py

Flask backend for the NileTech Policy Assistant.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.rag_chain import PolicyRAG

app = Flask(__name__)
# Enable CORS for all routes so your Streamlit client can communicate freely
CORS(app)

# Initialize the synchronized RAG engine
rag = PolicyRAG()


@app.route("/")
def home():
    """Returns basic API status information instead of a hardcoded redirect."""
    return jsonify(
        {
            "application": "NileTech Policy Assistant API",
            "status": "active",
            "version": "1.0.0"
        }
    ), 200


@app.route("/health", methods=["GET"])
def health_check():
    """Tells the frontend and test suite that the backend is fully active."""
    return {"status": "healthy"}, 200


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    if not data or "question" not in data:
        return jsonify(
            {
                "error": "Missing 'question' field."
            }
        ), 400

    question = data["question"]

    try:
        result = rag.generate(question)

        # If generate() already returns answer + sources
        if isinstance(result, dict):
            return jsonify(result), 200

        # Backward compatibility if generate() returns only a string
        return jsonify(
            {
                "answer": result,
                "sources": [],
            }
        ), 200

    except Exception as e:
        return jsonify(
            {
                "error": f"Internal RAG Chain Processing Exception: {str(e)}"
            }
        ), 500


if __name__ == "__main__":
    # Binding to 127.0.0.1 locally matches your frontend request configurations perfectly
    app.run(host="127.0.0.1", port=5000, debug=True)
