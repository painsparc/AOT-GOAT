"""Flask API + static UI for the AOT-GOAT prototype.   python app.py   |   gunicorn app:app"""
import json
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

load_dotenv()
import aotgoat  # noqa: E402
import evaluate  # noqa: E402
from llm import LLM  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")
app.json.sort_keys = False
_eval_cache = {}


def _llm(body):
    return LLM("offline") if body.get("use_llm") is False else LLM()


def _bad(msg, code=400):
    return jsonify(error=msg), code


def _check_message(body):
    m = body.get("message")
    if not isinstance(m, str) or not m.strip():
        return None, "`message` must be a non-empty string"
    if len(m) > 2000:
        return None, "`message` is limited to 2000 characters"
    return m.strip(), None


def _check_history(h):
    if h is None:
        return [], None
    if not isinstance(h, list) or len(h) > 200:
        return None, "`history` must be a list of at most 200 items ({sender, text} objects or 'S: ...' / 'R: ...' strings)"
    return h, None


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html", max_age=0)


@app.get("/api/health")
def health():
    llm = LLM()
    return jsonify(status="ok", provider=llm.provider, model=llm.model if llm.remote else None)


@app.get("/api/samples")
def samples():
    with open(os.path.join(HERE, "data", "sample_data.json"), encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.post("/api/analyze")
def analyze():
    """One recipient. Body: {message, history, sender_context?, meta?, recipient?, use_llm?}"""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _bad("Body must be a JSON object")
    msg, err = _check_message(body)
    if err:
        return _bad(err)
    hist, err = _check_history(body.get("history"))
    if err:
        return _bad(err)
    try:
        return jsonify(aotgoat.run(msg, hist, body.get("sender_context"), body.get("meta"), str(body.get("recipient", "recipient")), _llm(body)))
    except Exception as e:  # keep the demo alive and tell the user what went wrong
        return _bad(f"Pipeline error: {type(e).__name__}: {e}", 500)


@app.post("/api/compare")
def compare():
    """Same message, several recipient histories. Body: {message, sender_context?, recipients:[{id, history, meta?}], use_llm?}"""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _bad("Body must be a JSON object")
    msg, err = _check_message(body)
    if err:
        return _bad(err)
    recs = body.get("recipients")
    if not isinstance(recs, list) or not recs or len(recs) > 8:
        return _bad("`recipients` must be a list of 1-8 objects")
    llm, results = _llm(body), []
    try:
        for i, r in enumerate(recs):
            hist, err = _check_history(r.get("history"))
            if err:
                return _bad(err)
            results.append(aotgoat.run(msg, hist, body.get("sender_context"), r.get("meta"), str(r.get("id", f"recipient {i + 1}")), llm))
    except Exception as e:
        return _bad(f"Pipeline error: {type(e).__name__}: {e}", 500)
    return jsonify(message=msg, results=results)


@app.get("/api/evaluate")
def run_evaluation():
    """Runs the controlled evaluation with the deterministic offline analyzer (cached)."""
    if "report" not in _eval_cache:
        _eval_cache["report"] = evaluate.evaluate()
    return jsonify(_eval_cache["report"])


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")), debug=False)
