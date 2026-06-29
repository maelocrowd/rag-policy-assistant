"""
api.py

Flask backend for the NileTech Policy Assistant.
"""

from flask import Flask, jsonify, redirect, request

from src.rag_chain import PolicyRAG

app = Flask(__name__)

rag = PolicyRAG()


# @app.route("/")
# def home():
#     return jsonify(
#         {
#             "application": "NileTech Policy Assistant API",
#             "status": "running",
#         }
#     )

@app.route("/")
def home():
    return redirect("http://localhost:8501")

@app.route("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
        }
    )


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
            return jsonify(result)

        # Backward compatibility if generate() returns only a string
        return jsonify(
            {
                "answer": result,
                "sources": [],
            }
        )

    except Exception as e:

        return jsonify(
            {
                "error": str(e)
            }
        ), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)