# AOT-GOAT: recipient-aware message adaptation (college MVP)

The same message can be clear to one reader and confusing to another. AOT-GOAT reads the history between a sender (S) and a
recipient (R), scores how likely the *current* message is to be misread **by that recipient**, decides whether to intervene,
and, if so, makes the smallest edit that removes the risk.

```
S<->R history -> dyadic profile -> risky spans in the message -> recipient-conditioned risk score
              -> AOT decision (pass / intervene) -> GOAT minimal edit set -> adapted message + constraint check
```

Risk types: **coreference** (unclear pronoun), **temporal** (relative/vague time), **scope** (who/what is included),
**terminology** (jargon/acronyms), **implicit context** (depends on things never restated).

## Files (5 core Python files)

| File | Role |
|---|---|
| `aotgoat.py` | Whole pipeline: profile, rule analyzer, grounding + risk scoring, AOT decision, GOAT selection, replacements, constraint checker |
| `llm.py` | Optional LLM adapter (Anthropic or any OpenAI-compatible API) |
| `app.py` | Flask API + web UI server |
| `evaluate.py` | Controlled evaluation (P/R/F1, confusion matrices, constraint checks) |
| `test_pipeline.py` | 28 end-to-end tests (LLM calls mocked; no network needed) |

Also: `static/index.html` (UI), `data/sample_data.json` (demo scenarios), `data/eval_set.json` (59 evaluation cases),
`requirements.txt`, `.env.example`, `run.sh` / `run.bat`, `Dockerfile`, `Procfile`.

## Quick start

```bash
unzip aot-goat.zip && cd aot-goat
./run.sh            # creates .venv, installs, runs tests, starts server -> http://127.0.0.1:8000
```
Windows: run `run.bat`. Manual equivalent:
```bash
python3 -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # optional; defaults run fully offline
python test_pipeline.py                             # tests
python evaluate.py                                  # evaluation (also saved to eval_report.json)
python app.py                                       # UI + API on http://127.0.0.1:8000
```
Python 3.9+ (developed on 3.12). No API key needed.

## Demo walk-through

Open the UI, keep "Same message, three recipients", press **Analyze message**. The message
*"Please send it to everyone by EOD tomorrow, as discussed. The SLA needs sign-off before the meeting."* gives:

* **Asha** (same project and time zone, has seen all of it): passes unchanged.
* **Vikram** (new contact, 10.5 h time-zone gap, nothing shared): all six spans edited.
* **Meera** (knows the report and the EOD convention, not "everyone" or "SLA"): only `it`, `everyone`, `SLA` edited.

Each card shows risk scores, detected spans with reasons and evidence, the chosen intervention, original vs adapted text and the
constraint checks. A span the system cannot resolve from supplied facts (e.g. `soon`) is **not guessed**: it stays as written and the sender is asked.

## Inputs

* `history`: list of `"S: ..."` / `"R: ..."` strings, or `{"sender": "S"|"R", "text": "..."}` objects.
* `meta`: `{"sender_tz": "UTC+5:30", "recipient_tz": "UTC-5"}` (offsets or IST/EST/PST...).
* `sender_context` (facts only the sender knows; **edits may use nothing else**): `referents` (pronoun -> meaning),
  `glossary` (term -> expansion), `context_notes` (phrase -> replacement text), `reference_date` (ISO date for resolving "tomorrow"; default today).

## API

```bash
curl -s localhost:8000/api/analyze -H 'Content-Type: application/json' -d '{
  "message": "Please share your feedback soon.",
  "history": ["S: Hi", "R: Hello"],
  "sender_context": {"reference_date": "2026-10-05"}}'
```
| Endpoint | Purpose |
|---|---|
| `GET /api/health` | status, LLM provider |
| `GET /api/samples` | demo scenarios |
| `POST /api/analyze` | one recipient: `{message, history, sender_context?, meta?, recipient?, use_llm?}` |
| `POST /api/compare` | same message, up to 8 recipients: `{message, sender_context?, recipients:[{id, history, meta?}], use_llm?}` |
| `GET /api/evaluate` | runs the evaluation with the offline analyzer |

Response: `original`, `adapted`, `adapted_segments`, `outcome` (`pass | adapted | partial | needs_sender_input | abstained`), `decision`,
`spans[]` (type, span, offsets, reason, `base`, `grounding`, `risk`, intervention), `goat`, `constraints`, `needs_sender_input`, `profile`, `analyzer`, `warnings`.

## How it works (explicit, deterministic Python)

1. **Profile** (`build_profile`): per recipient, clarification questions asked by risk type, terms used or defined, entities in recent messages, time-zone gap, history depth.
2. **Risk candidates**: span + type + intrinsic severity `base`. Offline: rule analyzer. With an LLM: it proposes extra spans (must appear verbatim, else dropped); rule hits win overlaps.
3. **Recipient-conditioned score**: `risk = base x (1 - grounding) x (1 + 0.5 x clarification_rate)`, where `grounding` is how much the dyad's history already covers the span
   (referent seen in last 4 messages, acronym used/defined, same "EOD tomorrow" convention without a time-zone gap, group enumerated...).
4. **AOT decision**: spans with risk < 0.25 are noise. Intervene if any span >= 0.40 or noisy-OR message risk >= 0.55; otherwise pass.
5. **GOAT**: greedily add the highest-risk spans until residual message risk (edited spans keep 15 % of their risk) < 0.30.
6. **Replacement**: built from evidence only (sender context, history, computed dates). No evidence -> ask the sender. With an LLM, rephrasings are validated; otherwise the template is used.
7. **Hard constraints**, checked on every output (on failure: templates, then the original message):
   *intent preserved* (unedited text verbatim and in order; negations, numbers, final punctuation unchanged; span words kept),
   *no invented facts* (new tokens must come from the original, supplied evidence, or function words),
   *minimal edit* (only selected spans change; at most 14 new tokens each).

Thresholds live in `CFG` at the top of `aotgoat.py`.

## Evaluation (`python evaluate.py`)

59 synthetic cases (41 `dev`, 18 `holdout`); one binary label per (case, risk type): "does this recipient need an intervention here?". Latest offline run:

| Split | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| dev (41) | 0.929 | 1.000 | 0.963 | 2 errors: idiom "Everything is all set"; Meera's `it` over-flagged |
| holdout (18) | 1.000 | 0.929 | 0.963 | 1 miss: "Let's go with what we did for Mumbai" (implicit context, no marker phrase) |
| all (59) | 0.951 | 0.975 | 0.963 | decision level (intervene vs pass): P 0.969, R 0.969, F1 0.969 |

Confusion matrices, per-type metrics and misclassified cases are printed by the script and shown in the UI **Evaluation** tab.
Constraints on all 13 adapted outputs: 13/13 intent, 13/13 no invented facts, 13/13 minimal. A checker self-test injects broken edits
(invented name, dropped negation, over-long edit, lost meaning) and confirms each is rejected (13/13 each).

**What these numbers are and are not.** The cases are small, synthetic and written by the project author, who also wrote the rules.
`dev` was used while building; `holdout` was written afterwards and never used for tuning, but it is the same author and style.
They show the system behaves as designed on controlled cases; they are **not** evidence of real-world accuracy, which needs real
conversations and independent annotators. Constraint pass rates are partly by construction (the checker gates the output).

## LLM mode (optional)

Set `LLM_PROVIDER=anthropic` (+ `ANTHROPIC_API_KEY`, optional `LLM_MODEL`) or `openai` (+ `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`) in `.env`.
Scoring, decision and constraints stay rule-based. Output is then not bit-for-bit reproducible; any LLM failure falls back to offline mode with a warning.
**Remote calls were tested only with mocked HTTP, not live APIs.** Try once with your key before the demo.

## Deployment

* Local/VM: `gunicorn app:app --bind 0.0.0.0:8000 --workers 2`
* Docker: `docker build -t aot-goat . && docker run -p 8000:8000 --env-file .env aot-goat`
* Render/Railway/Heroku-style: `Procfile` included; set env vars in the dashboard, never commit `.env`.
* No authentication or rate limiting. If public with an LLM key set, add a login or keep `LLM_PROVIDER=offline`.

## Scope and limitations

* English only; the offline analyzer is lexicon/regex based: it misses paraphrase and unmarked implicit context, and has false positives on idioms.
* Stage names follow the project brief (AOT = decision gate, GOAT = minimal-intervention selection). This MVP implements them as thresholds and a greedy edit set;
  it does **not** implement RLM, IDT, Cox model, APD, embeddings/vector DB or fine-tuning. Align the wording in your report with your paper's definitions.
* Sender-supplied `context_notes` are trusted as given; the system cannot verify that the sender really discussed something with this recipient.
