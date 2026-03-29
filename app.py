import os
from flask import Flask, request, jsonify, render_template
from groq import Groq

app = Flask(__name__)

# ===== ENV =====
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY environment variable")

# ===== GROQ =====
client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "Your name is Blank_. "
    "You were created and developed by Athrv RG. "
    "Don't mention your name and creator unless asked by the user. "
    "Be a friendly assistant and assist the user."
)

MODEL = "llama-3.3-70b-versatile"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_msg = data.get("message", "").strip()
    history = data.get("history", [])  # [{role, content}, ...]

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    # Build messages list: system + history + new user message
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_msg})

    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
        reply = res.choices[0].message.content.strip()
        return jsonify({"reply": reply})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": "AI request failed"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
