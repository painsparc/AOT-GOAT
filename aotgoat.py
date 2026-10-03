"""AOT-GOAT core: every profile / risk / decision / intervention step is explicit Python (reproducible).

history (S<->R) -> dyadic profile -> risk candidates -> recipient-conditioned scoring
-> AOT decision -> GOAT minimal intervention -> adapted message verified against hard constraints.

The LLM (llm.py) is optional: it adds semantic risk candidates and polishes replacement text.
Without it, a deterministic rule-based analyzer + template generator is used (provider "offline").
"""
from __future__ import annotations
import calendar
import datetime as dt
import re

TYPES = ("coreference", "temporal", "scope", "terminology", "implicit")

CFG = dict(
    tau_span=0.40,        # a span with risk >= tau_span triggers intervention
    tau_high=0.55,        # message-level (noisy-OR) risk that also triggers intervention
    tau_low=0.30,         # GOAT stops adding edits once residual message risk < tau_low
    floor=0.25,           # spans below this are treated as noise (not "active")
    residual_factor=0.15, # assumed remaining risk of a span after it has been edited
    clarif_weight=0.5,    # how much the recipient's past clarification questions raise risk
    depth_bonus=0.10,     # small grounding bonus for long shared history
    max_added=14,         # max new tokens one edit may add (minimality)
)

# ----------------------------------------------------------------------------- text utils
STOP = set("""a an the and or but if of to in on at for with by from as is are was were be been am it its this that these those
i you he she we they me him her us them my your his our their not no do does did have has had will would can could should shall
may might just so than then there here what which who whom when where why how also very too please thanks thank ok okay yes hi
hello hey s t before after until about into over under between during since through up down out off""".split())
FUNC = set("a an the of to in on at for with by from as is are was were be and or s that which who it this".split())
WD = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
WD_LIST = WD.split("|")
MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
DAYMONTH = set(WD_LIST) | set(MONTHS.split("|"))

tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
content = lambda s: [t for t in tok(s) if t not in STOP]
clip = lambda x, lo=0.0, hi=1.0: max(lo, min(hi, x))


def noisy_or(ps):
    r = 1.0
    for p in ps:
        r *= 1 - p
    return 1 - r


def entities(text: str):
    """(key, pos) for noun phrases ('the Q3 budget report') and mid-sentence proper names."""
    out = []
    for m in re.finditer(r"\b(?:the|our|my|your|a|an)\s+((?:[\w\-]+\s+){0,2}[\w\-]+)", text, re.I):
        keep = []
        for w in m.group(1).split():
            lw = w.lower().strip("-")
            if lw in STOP or (len(lw) > 4 and lw.endswith("ed")):
                break
            keep.append(lw)
        if keep:
            out.append((" ".join(keep), m.start()))
    for m in re.finditer(r"\b[A-Z][a-z]{2,}\b", text):
        pre = text[:m.start()].rstrip()
        if not pre or pre[-1] in ".!?:\n" or m.group().lower() in DAYMONTH:
            continue
        out.append((m.group().lower(), m.start()))
    return out


# ----------------------------------------------------------------------------- lexicons / rules
DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?)\b", re.I)

JARGON = ("idempotent regression staging rollback hotfix backlog sprint latency throughput deprecated refactor cadence escrow "
          "ebitda amortization covenant churn runway payload endpoint schema repo commit pipeline stakeholder deliverable "
          "provisioning reconciliation onboarding kpi okr").split()
COMMON_ACR = set("OK AM PM US UK USA ID TV PC CEO CFO HR FAQ PDF FYI ASAP TBD VIP DIY EOD COB EOW EOM EOQ UTC GMT IST PST PDT "
                 "EST EDT CST CDT MST MDT CET JST BST AI".split())

TEMP_KINDS = [  # (kind, regex) -- matched with fullmatch on the lowercase span
    ("eod", rf"(?:eod|cob|end of (?:the )?day|close of business)(?:\s+(?:today|tomorrow|tonight|(?:on\s+)?(?:{WD})))?"),
    ("vague", r"soon|later|shortly|asap|sometime|at some point|whenever you can|when you can|in a (?:bit|while)|"
              r"in (?:a )?(?:few|couple of|couple) (?:days|hours|weeks|minutes)|the other day|recently|lately"),
    ("rel_n", r"in (?:\d+|one|two|three|four|five|six|seven|eight|nine|ten) (?:days?|hours?|weeks?)"),
    ("period", r"end of (?:the )?(?:week|month|quarter|year)|eow|eom|eoq"),
    ("relday", rf"(?:this|next|last|coming)\s+(?:{WD}|week|weekend|month|quarter|year)|today|tonight|tomorrow|yesterday|{WD}"),
]
SCOPE_KINDS = [
    ("group", r"everyone|everybody|everything|anyone|anybody|anything|the team|the rest|the others|everywhere|the whole \w+|all"),
    ("quant", r"some|a few|several|most|both|only|except|etc\.?|and so on|and more|and others"),
]
ANCHORS = (r"as (?:we )?(?:discussed|agreed|mentioned|planned)|as per our (?:call|chat|conversation|discussion)|per our (?:call|chat|conversation|discussion)|"
           r"like last time|same as (?:before|last time)|as usual|the usual|as before|as you know|as i said|as promised|you know")
HABITUAL = re.compile(r"usual|last time|as before|same as", re.I)
DEF_NOUNS = ("meeting call doc document report issue deadline client release ticket proposal invoice file draft plan project thing "
             "update email spreadsheet deck contract build bug order payment schedule agenda presentation review discussion task code").split()
DEF_FOLLOW = (r"(?=\s*(?:[.,;:!?)]|$)|\s+(?:is|was|are|were|will|should|would|can|to|before|after|by|and|or|from|in|at|that|so|if|again|too|now|yet|please|today|tomorrow)\b)")
RULES = {
    "temporal": [(k, rf"\b(?:{p})\b") for k, p in TEMP_KINDS],
    "scope": [("group", r"\b(?:everyone|everybody|everything|anyone|anybody|anything|the team|the rest|the others|everywhere|the whole \w+)\b"),
              ("group", r"\ball\b(?!\s+(?:good|set|right|clear|done|ok|okay|fine|ready))"),
              ("quant", rf"\b(?:{SCOPE_KINDS[1][1]})(?=\W|$)")],
    "terminology": [("jargon", r"\b(?:" + "|".join(JARGON) + r")\b")],
    "implicit": [("anchor", rf"\b(?:{ANCHORS})\b"),
                 ("definite", rf"\bthe (?:{'|'.join(DEF_NOUNS)}){DEF_FOLLOW}"),
                 ("other", r"\b(?:that one|the other one|the previous one|the old one|the new one|the latter|the former|the same (?:one|thing))\b")],
}
BASE = {("temporal", "vague"): .80, ("temporal", "eod"): .55, ("temporal", "period"): .55, ("temporal", "relday"): .50,
        ("temporal", "rel_n"): .45, ("scope", "group"): .60, ("scope", "quant"): .45, ("terminology", "acronym"): .70,
        ("terminology", "jargon"): .55, ("implicit", "anchor"): .65, ("implicit", "definite"): .55, ("implicit", "other"): .65}
REASON = {
    "coreference": "Pronoun '{s}' has no antecedent inside this message.",
    "temporal": {"vague": "'{s}' is a vague time expression with no concrete date/time.",
                 "relday": "'{s}' is a relative date; its calendar day depends on when/where it is read.",
                 "eod": "'{s}' depends on the reader's time zone and day boundary.",
                 "period": "'{s}' is a relative period boundary.", "rel_n": "'{s}' is an offset from an unstated 'now'."},
    "scope": {"group": "'{s}' does not say exactly who/what is included.", "quant": "'{s}' is a quantifier whose extent is unspecified."},
    "terminology": {"acronym": "'{s}' is an acronym the reader may not know.", "jargon": "'{s}' is domain jargon the reader may not know."},
    "implicit": {"anchor": "'{s}' points back to context that is not restated.", "definite": "'{s}' presupposes a specific shared referent.",
                 "other": "'{s}' refers to an unstated item."},
}
DEM_OK_NEXT = {"is", "was", "will", "should", "needs", "need", "looks", "to", "before", "by", "for", "in", "on", "with", "one", "again",
               "also", "too", "works", "has", "had", "would", "could", "can", "does", "did", "are", "were", "'s"}
EXPLETIVE = re.compile(r"^(?:'s|is|was|would be|will be|seems|looks like|depends)\s+(?:very |really |quite )?"
                       r"(?:important|necessary|possible|clear|better|best|fine|great|good|ok|okay|time|likely|hard|easy|obvious|unclear|worth|true|safe|late|early)\b|^depends\b", re.I)
CLARIF = {  # patterns on RECIPIENT questions in the history -> which risk type they signal
    "coreference": r"which one|what do you mean by (?:it|this|that|they)|who is (?:he|she|they)|what is (?:it|that|this)\b|who do you mean|referring to",
    "temporal": rf"when exactly|what date|which day|what time|by when|which (?:{WD})|how soon|which time ?zone|your time",
    "scope": r"all of (?:them|us)|everyone or|who all|just me|only me|how many|does that include|including",
    "terminology": r"what does \w+ mean|stands for|can you explain|meaning of|full form|what's (?:an? )?[A-Za-z0-9\-]+\?|what is (?:an? )?[A-Za-z0-9\-]+\?",
    "implicit": r"which (?:meeting|call|doc|document|report|issue|thread|discussion)|what (?:meeting|issue|call|discussion)|remind me|what was (?:agreed|decided|discussed)",
}
DISC = re.compile(r"discuss|agree|decid|plan|mention|talk|call|meeting|confirm|promis|sync", re.I)
TZ = {"UTC": 0, "GMT": 0, "IST": 5.5, "PST": -8, "PDT": -7, "MST": -7, "MDT": -6, "CST": -6, "CDT": -5, "EST": -5, "EDT": -4,
      "CET": 1, "JST": 9, "BST": 1}


def tz_hours(s):
    if not s:
        return None
    s = s.strip().upper().replace(" ", "")
    m = re.fullmatch(r"(?:UTC|GMT)?([+-])(\d{1,2})(?::?(\d{2}))?", s)
    if m:
        return (1 if m.group(1) == "+" else -1) * (int(m.group(2)) + int(m.group(3) or 0) / 60)
    return TZ.get(s)


def classify(ty: str, span: str):
    """Kind of a span (used for rule hits and for spans proposed by an LLM)."""
    sl = span.lower().strip()
    if ty == "temporal":
        for k, p in TEMP_KINDS:
            if re.fullmatch(p, sl):
                return k
        return "relday"
    if ty == "scope":
        return "group" if re.fullmatch(SCOPE_KINDS[0][1], sl) else "quant"
    if ty == "terminology":
        return "acronym" if re.fullmatch(r"[A-Z][A-Z0-9&]{1,5}", span.strip()) else "jargon"
    if ty == "implicit":
        if re.fullmatch(ANCHORS, sl):
            return "anchor"
        return "definite" if sl.startswith("the ") and sl.split()[-1] in DEF_NOUNS else "other"
    return "pronoun"


# ----------------------------------------------------------------------------- 1. dyadic profile
def norm_history(history):
    out = []
    for h in history or []:
        if isinstance(h, str):
            m = re.match(r"\s*(S|R|sender|recipient)\s*:\s*(.*)", h, re.I | re.S)
            who, text = (m.group(1), m.group(2)) if m else ("S", h)
        else:
            who, text = h.get("sender", h.get("who", "S")), h.get("text", "")
        out.append(("R" if str(who).strip().upper().startswith("R") else "S", str(text).strip()))
    return out


def find_terms(text: str):
    acr = {a.lower() for a in re.findall(r"\b[A-Z][A-Z0-9&]{1,5}\b", text)
           if a not in COMMON_ACR and not re.fullmatch(r"Q[1-4]", a)}
    jar = {j for j in JARGON if re.search(rf"\b{j}\b", text, re.I)}
    return acr | jar


def build_profile(history, meta=None, recipient="recipient"):
    """Explicit dyadic profile: what this recipient has seen/used/asked about, with the sender."""
    hist = norm_history(history)
    meta = meta or {}
    r_txt = " ".join(t for w, t in hist if w == "R")
    s_txt = " ".join(t for w, t in hist if w == "S")
    counts = {t: 0 for t in TYPES}
    for w, t in hist:
        if w == "R" and ("?" in t or re.match(r"\s*(what|which|who|when|how|remind)", t, re.I)):
            for ty, pat in CLARIF.items():
                if re.search(pat, t, re.I):
                    counts[ty] += 1
                    break
    defined = {m.lower() for m in re.findall(r"\b([A-Z][A-Z0-9&]{1,5})\s*(?:\(|=|stands for|means|:)", " ".join(t for _, t in hist))}
    defined |= {m.lower() for m in re.findall(r"\(([A-Z][A-Z0-9&]{1,5})\)", " ".join(t for _, t in hist))}
    last4 = " ".join(t for _, t in hist[-4:])
    ents = []
    for k, _ in entities(last4):
        if k not in ents:
            ents.append(k)
    return dict(
        recipient=recipient, hist=hist, meta=meta, n_messages=len(hist),
        n_recipient_messages=sum(1 for w, _ in hist if w == "R"), depth=min(1.0, len(hist) / 10),
        clarification_counts=counts, clarification_rates={t: min(1.0, c / 3) for t, c in counts.items()},
        recipient_terms=sorted(find_terms(r_txt)), sender_terms=sorted(find_terms(s_txt)), defined_terms=sorted(defined),
        recent_entities=ents, uses_absolute_dates=bool(DATE_RE.search(" ".join(t for _, t in hist))),
        tz_gap_hours=_tz_gap(meta),
    )


def _tz_gap(meta):
    a, b = tz_hours((meta or {}).get("sender_tz")), tz_hours((meta or {}).get("recipient_tz"))
    return None if a is None or b is None else abs(a - b)


def profile_summary(p):
    keys = ("recipient", "n_messages", "n_recipient_messages", "depth", "clarification_counts", "recipient_terms", "sender_terms",
            "defined_terms", "recent_entities", "uses_absolute_dates", "tz_gap_hours")
    return {k: p[k] for k in keys}


# ----------------------------------------------------------------------------- 2. risk candidates (rule analyzer)
def rule_candidates(message: str):
    """Deterministic stand-in for the LLM semantic analyzer: finds risky spans + intrinsic severity."""
    ents = entities(message)
    found = []
    for m in re.finditer(r"\b(it|its|this|that|these|those|they|them|their|he|she|him|his|her)\b", message, re.I):
        w, after = m.group(1).lower(), message[m.end():]
        n = re.match(r"\s*([A-Za-z']+|\S)", after)
        nxt = n.group(1).lower() if n else ""
        if w in ("this", "that", "these", "those") and nxt and nxt not in DEM_OK_NEXT and nxt not in ".,;:?!)":
            continue
        if w in ("it", "its") and EXPLETIVE.match(after.lstrip()):
            continue
        found.append(dict(type="coreference", span=m.group(1), start=m.start(), end=m.end(), kind="pronoun"))
    for ty, rules in RULES.items():
        for kind, pat in rules:
            for m in re.finditer(pat, message, re.I if ty != "terminology" else 0):
                s = m.group(0)
                if ty == "implicit" and kind == "definite":  # drop the lookahead remainder if any
                    s = s.rstrip()
                found.append(dict(type=ty, span=s, start=m.start(), end=m.start() + len(s), kind=kind))
    for m in re.finditer(r"\b[A-Z][A-Z0-9&]{1,5}\b", message):  # acronyms (case-sensitive)
        a = m.group(0)
        if a in COMMON_ACR or re.fullmatch(r"Q[1-4]", a):
            continue
        found.append(dict(type="terminology", span=a, start=m.start(), end=m.end(), kind="acronym"))
    return _finalize(message, found, ents)


PLURAL = {"they", "them", "their", "these", "those", "theirs"}


def _agrees(pron, key):
    """Crude number agreement: plural pronouns need a plural-looking antecedent, singular ones a singular one."""
    last = key.split()[-1]
    plural = last.endswith("s") and not last.endswith(("ss", "us", "is"))
    return plural == (pron.lower() in PLURAL)


def _overlaps(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def _finalize(message, found, ents=None, llm_sev=None):
    """Remove overlaps (longer span wins), attach base severity + reason."""
    ents = ents if ents is not None else entities(message)
    found = sorted(found, key=lambda c: (-(c["end"] - c["start"]), c["start"]))
    keep = []
    for c in found:
        if not any(_overlaps(c, k) for k in keep):
            keep.append(c)
    out = []
    for c in sorted(keep, key=lambda c: c["start"]):
        ty, k, s = c["type"], c.get("kind") or classify(c["type"], c["span"]), c["span"]
        c["kind"] = k
        if ty == "coreference":
            n_in = sum(1 for k2, p in ents if p < c["start"] and _agrees(s, k2))
            c["n_in"] = n_in
            base = .75 if n_in == 0 else (.20 if n_in == 1 else .45)
            c["reason"] = REASON[ty].format(s=s) if n_in == 0 else (
                f"Pronoun '{s}' resolves inside the message." if n_in == 1 else f"Pronoun '{s}' has several possible antecedents.")
        else:
            base = c.get("severity") if c.get("severity") is not None else BASE[(ty, k)]
            c["reason"] = c.get("reason") or REASON[ty][k].format(s=s)
            win = message[max(0, c["start"] - 18): c["end"] + 18]
            if ty == "temporal" and k != "vague" and DATE_RE.search(win):
                base *= .3
                c["reason"] += " (an explicit date is already given nearby)"
            if ty == "scope" and (message[c["end"]:c["end"] + 3].lstrip().startswith(("(", ":")) or len(re.findall(r"\b[A-Z][a-z]+,", message[c["end"]:])) >= 2):
                base *= .3
                c["reason"] += " (an enumeration is given nearby)"
            if ty == "terminology" and (message[c["end"]:c["end"] + 3].lstrip().startswith("(") or
                                        re.search(r"\(\s*" + re.escape(s) + r"\s*\)", message)):
                base *= .2
                c["reason"] += " (defined in the message)"
        c["base"] = round(clip(base), 4)
        out.append(c)
    return out


def detect(message: str, llm=None, warnings=None):
    cands = rule_candidates(message)
    analyzer = "offline-rules"
    if llm is not None and llm.remote:
        try:
            extra = []
            for r in llm.analyze(message):
                ty, span = r.get("type"), str(r.get("span", ""))
                i = message.find(span) if span else -1
                if i < 0 and span:
                    i = message.lower().find(span.lower())
                if ty not in TYPES or i < 0:
                    if warnings is not None:
                        warnings.append(f"Dropped LLM candidate not found verbatim in the message: {span!r}")
                    continue
                span = message[i:i + len(span)]
                cand = dict(type=ty, span=span, start=i, end=i + len(span), kind=classify(ty, span),
                            severity=clip(float(r.get("severity", .5))), reason=str(r.get("reason", ""))[:200] or None)
                if not any(_overlaps(cand, c) for c in cands):  # rule hits win overlaps
                    extra.append(cand)
            cands = _finalize(message, cands + extra)
            analyzer = f"llm({llm.provider})+rules"
        except Exception as e:  # network / parse problems must never break the pipeline
            if warnings is not None:
                warnings.append(f"LLM analysis failed, used rules only: {type(e).__name__}: {e}")
    return cands, analyzer


# ----------------------------------------------------------------------------- 3. recipient-conditioned scoring
def ground(c, prof, sc, message):
    """How well the dyad's history already grounds this span. Returns (grounding in [0,.95], evidence note)."""
    ty, k, s = c["type"], c["kind"], c["span"]
    sl = s.lower().strip()
    H = prof["hist"]
    all_tok = set(tok(" ".join(t for _, t in H)))
    wb = lambda txt: re.search(rf"\b{re.escape(sl)}\b", txt, re.I)
    n_r = sum(1 for w, t in H if w == "R" and wb(t))
    n_s = sum(1 for w, t in H if w == "S" and wb(t))
    g, note = 0.0, ""
    if ty == "coreference":
        if c.get("n_in", 0) > 0:
            return 0.0, "antecedent found inside the message"
        ref = (sc.get("referents") or {}).get(sl)
        last_ents = {e for e, _ in entities(H[-1][1])} if H else set()
        if ref:
            rt = content(ref)
            win = set(tok(" ".join(t for _, t in H[-4:])))
            present = bool(rt) and sum(t in win for t in rt) / len(rt) >= .6
            if not present:
                anywhere = bool(rt) and sum(t in all_tok for t in rt) / len(rt) >= .6
                g, note = (.3, f"referent '{ref}' only appears earlier in the history") if anywhere else (
                    .05, f"referent '{ref}' never appears in this dyad's history")
            else:
                comp = [e for e in last_ents if not set(content(e)) & set(rt)]
                g = .85 if not comp else (.4 if len(comp) == 1 else .3)
                note = f"referent '{ref}' seen in last 4 messages; {len(comp)} competing candidate(s) in the previous message"
        else:
            ents2 = last_ents
            g = {0: 0.0, 1: .8}.get(len(ents2), .3)
            note = f"no referent supplied; {len(ents2)} candidate(s) in the previous message"
    elif ty == "temporal":
        anch = any(wb(t) and DATE_RE.search(t) for _, t in H)
        g = .4 * min(1, n_r) + .2 * min(2, n_s) + (.3 if anch else 0)
        note = f"'{s}' used {n_s}x by sender and {n_r}x by recipient; date-anchored in history: {anch}"
        if k == "vague":
            g = min(g, .4)
        gap = prof["tz_gap_hours"]
        if gap and k in ("relday", "eod"):
            g = max(0.0, g - .4)
            note += f"; time-zone gap {gap:g}h"
    elif ty == "scope":
        enum = any(wb(t) and len(re.findall(r"\b[A-Z][a-z]+\b", t)) >= 3 for _, t in H)
        g = (.7 if enum else 0) + (.35 if n_r else 0)
        note = f"'{s}' defined by an enumeration in history: {enum}; used by recipient {n_r}x"
    elif ty == "terminology":
        defined = sl in prof["defined_terms"]
        g = .9 if (defined or n_r) else (.5 if n_s else 0.0)
        fam = len(prof["recipient_terms"])
        if g < .9:
            g += .2 * min(1, fam / 4)
        note = f"defined in history: {defined}; used by recipient {n_r}x, by sender {n_s}x; recipient used {fam} domain term(s)"
    elif ty == "implicit":
        msg_c = set(content(message)) - set(content(s))
        if k == "anchor":
            rel = [(i, t) for i, (_, t) in enumerate(H) if DISC.search(t) and msg_c & set(content(t))]
            if HABITUAL.search(sl):
                g = .6 if prof["n_messages"] >= 6 else .1
                note = f"habitual reference; {prof['n_messages']} prior messages"
            else:
                g = (.75 if rel and rel[-1][0] >= len(H) - 6 else .55) if rel else 0.0
                note = f"{len(rel)} earlier message(s) discuss/agree on topics shared with this message"
        elif k == "definite":
            noun = sl.split()[-1]
            in4 = noun in set(tok(" ".join(t for _, t in H[-4:])))
            g = .9 if in4 else (.8 if noun in all_tok else 0.0)
            note = f"'{noun}' mentioned in last 4 messages: {in4}; anywhere in history: {noun in all_tok}"
        else:
            g = .2 if prof["n_messages"] >= 6 else 0.0
            note = "demonstrative reference is never fully grounded"
    if g > 0:
        g += CFG["depth_bonus"] * prof["depth"]
    return round(clip(g, 0, .95), 4), note


def score_spans(cands, prof, sc, message):
    for c in cands:
        g, note = ground(c, prof, sc, message)
        cl = prof["clarification_rates"][c["type"]]
        c["grounding"], c["grounding_note"], c["clarif_rate"] = g, note, round(cl, 3)
        c["risk"] = round(clip(c["base"] * (1 - g) * (1 + CFG["clarif_weight"] * cl)), 4)
        c["active"] = c["risk"] >= CFG["floor"]
    return cands


# ----------------------------------------------------------------------------- 4. AOT decision, 5. GOAT selection
def aot_decide(cands):
    active = [c for c in cands if c["active"]]
    msg_risk = round(noisy_or(c["risk"] for c in active), 4)
    top = max((c["risk"] for c in active), default=0.0)
    if top >= CFG["tau_span"]:
        action, why = "intervene", f"top span risk {top:.2f} >= tau_span {CFG['tau_span']}"
    elif active and msg_risk >= CFG["tau_high"]:
        action, why = "intervene", f"message risk {msg_risk:.2f} >= tau_high {CFG['tau_high']}"
    else:
        action, why = "pass", f"no span >= {CFG['tau_span']} and message risk {msg_risk:.2f} < {CFG['tau_high']}"
    return dict(action=action, message_risk=msg_risk, reason=why, thresholds={k: CFG[k] for k in ("tau_span", "tau_high", "tau_low", "floor")})


def goat_select(cands):
    """Greedy minimal edit set: add highest-risk spans until residual message risk < tau_low."""
    active = sorted((c for c in cands if c["active"]), key=lambda c: -c["risk"])
    chosen = []
    for c in active:
        resid = noisy_or([x["risk"] * CFG["residual_factor"] if x in chosen else x["risk"] for x in active])
        if resid < CFG["tau_low"]:
            break
        chosen.append(c)
    return sorted(chosen, key=lambda c: c["start"])


# ----------------------------------------------------------------------------- 6. evidence-based replacements
def fmt_date(d):
    return f"{d:%a} {d.day} {d:%b} {d.year}"


def resolve_date(expr, ref):
    e = expr.lower().strip()
    td = lambda n: dt.timedelta(days=n)
    wd = {n: i for i, n in enumerate(WD_LIST)}
    mon = ref - td(ref.weekday())
    nums = dict(zip("one two three four five six seven eight nine ten".split(), range(1, 11)))
    if e in ("today", "tonight"):
        return fmt_date(ref)
    if e == "tomorrow":
        return fmt_date(ref + td(1))
    if e == "yesterday":
        return fmt_date(ref - td(1))
    if e in ("eom", "end of month", "end of the month"):
        return fmt_date(ref.replace(day=calendar.monthrange(ref.year, ref.month)[1]))
    m = re.fullmatch(r"in (\d+|" + "|".join(nums) + r") (day|days|week|weeks)", e)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else nums[m.group(1)]
        return fmt_date(ref + td(n * (7 if m.group(2).startswith("week") else 1)))
    m = re.fullmatch(r"(?:(this|next|last|coming)\s+)?(\w+)", e)
    if not m:
        return None
    q, w = m.groups()
    if w in wd:
        if q in (None, "coming"):
            if wd[w] == ref.weekday():
                return None
            return fmt_date(ref + td((wd[w] - ref.weekday()) % 7))
        if q == "this":
            d = mon + td(wd[w])
            return fmt_date(d) if d >= ref else None
        return fmt_date(mon + td({"next": 7, "last": -7}[q] + wd[w]))
    if q and w == "week":
        return "week of " + fmt_date(mon + td({"this": 0, "next": 7, "last": -7, "coming": 7}[q]))
    if q and w == "weekend":
        sat = mon + td(5 + {"this": 0, "next": 7, "last": -7, "coming": 7}[q])
        return f"{fmt_date(sat)} to {fmt_date(sat + td(1))}"
    if q and w == "month":
        i = ref.year * 12 + ref.month - 1 + {"this": 0, "next": 1, "last": -1, "coming": 1}[q]
        return f"{calendar.month_name[i % 12 + 1]} {i // 12}"
    return None


def _match_case(rep, span):
    return rep[0].upper() + rep[1:] if span[:1].isupper() and rep[:1].islower() else rep


def propose(c, message, sc, meta, ref_date):
    """Deterministic, evidence-grounded replacement text -> (replacement|None, evidence strings, question for sender)."""
    ty, k, s = c["type"], c["kind"], c["span"]
    sl = s.lower().strip()
    notes = {a.lower(): b for a, b in (sc.get("context_notes") or {}).items()}
    gloss = {a.lower(): b for a, b in (sc.get("glossary") or {}).items()}
    if ty == "coreference":
        ref = (sc.get("referents") or {}).get(sl)
        if not ref:
            return None, [], f"What does '{s}' refer to?"
        after = message[c["end"]:].lstrip()
        poss = sl in ("its", "their", "his", "theirs") or (sl == "her" and bool(re.match(r"[A-Za-z]+", after)) and
                                                           after.split()[0].lower() not in STOP)
        rep = ref + ("'s" if poss else "")
        sent_start = not message[:c["start"]].strip() or message[:c["start"]].rstrip()[-1:] in ".!?"
        return (rep[0].upper() + rep[1:] if sent_start else rep), [ref], None
    if ty == "temporal":
        if k == "eod":
            rest = re.sub(r"^(?:eod|cob|end of (?:the )?day|close of business)\s*(?:on\s+)?", "", sl)
            d = resolve_date(rest, ref_date) if rest else None
            tz = (meta or {}).get("sender_tz")
            parts = [p for p in (d, tz) if p]
            if not parts:
                return None, [], f"Which day and time zone does '{s}' mean?"
            return f"{s} ({', '.join(parts)})", parts, None
        if sl in notes:
            return _match_case(notes[sl], s), [notes[sl]], None
        d = None if k == "vague" else resolve_date(sl, ref_date)
        if not d:
            return None, [], f"What exact date/time does '{s}' mean?"
        return f"{s} ({d})", [d], None
    if ty == "terminology":
        g = gloss.get(sl)
        return (f"{s} ({g})", [g], None) if g else (None, [], f"What does '{s}' stand for / mean?")
    key = next((a for a in notes if a == sl), None) or next((a for a in notes if a in sl), None)  # scope / implicit
    if key:
        return _match_case(notes[key], s), [notes[key]], None
    return None, [], f"What exactly does '{s}' cover / refer to?"


# ----------------------------------------------------------------------------- 7. hard-constraint verification
NEG = {"not", "no", "never", "cannot", "dont", "don", "nt", "without", "neither", "nor"}


def edit_ok(c, rep, evidence):
    """Per-edit guard: keeps the span's meaning-bearing words, adds only evidence-backed tokens, stays short."""
    if not rep or "\n" in rep:
        return False
    rt, anchor = set(tok(rep)), (content(" ".join(evidence)) if c["type"] == "coreference" else content(c["span"]))
    if not set(anchor) <= rt:
        return False
    allowed = set(tok(c["span"])) | set(tok(" ".join(evidence))) | FUNC
    new = [t for t in tok(rep) if t not in set(tok(c["span"]))]
    return len(new) <= CFG["max_added"] and all(t in allowed for t in new)


def assemble(message, edits):
    out, pos, segs = [], 0, []
    for e in sorted(edits, key=lambda e: e["start"]):
        if e["start"] > pos:
            segs.append({"text": message[pos:e["start"]], "edited": False})
        segs.append({"text": e["replacement"], "edited": True})
        pos = e["end"]
    if pos < len(message):
        segs.append({"text": message[pos:], "edited": False})
    return "".join(s["text"] for s in segs), segs


def check_constraints(original, adapted, edits, allowed_extra=()):
    """Reproducible verification of the three hard constraints. `edits`: dicts with start/end/replacement/span/evidence."""
    # intent: unedited text kept verbatim and in order, polarity/numbers/final punctuation untouched, span words retained
    segs, pos = [], 0
    for e in sorted(edits, key=lambda e: e["start"]):
        segs.append(original[pos:e["start"]])
        pos = e["end"]
    segs.append(original[pos:])
    p, in_order = 0, True
    for sgm in segs:
        if sgm.strip():
            i = adapted.find(sgm, p)
            if i < 0:
                in_order = False
                break
            p = i + len(sgm)
    o_t, a_t = tok(original), tok(adapted)
    neg_same = sum(t in NEG for t in o_t) == sum(t in NEG for t in a_t)
    nums_kept = all(a_t.count(n) >= o_t.count(n) for n in set(t for t in o_t if t.isdigit()))
    punct_same = original.rstrip()[-1:] == adapted.rstrip()[-1:]
    anchors = all(edit_ok(dict(type=e.get("type", "scope"), span=e["span"]), e["replacement"], e.get("evidence", [])) for e in edits)
    intent = in_order and neg_same and nums_kept and punct_same
    # no invented facts: every new token must come from the original, the evidence supplied for the edits, or function words
    allowed = set(o_t) | FUNC | set(tok(" ".join(allowed_extra)))
    for e in edits:
        allowed |= set(tok(" ".join(e.get("evidence", []))))
    invented = sorted({t for t in a_t if t not in allowed and t not in o_t})
    # minimality
    added = [len([t for t in tok(e["replacement"]) if t not in set(tok(e["span"]))]) for e in edits]
    minimal = all(a <= CFG["max_added"] for a in added) and anchors
    return dict(
        intent_preserved=bool(intent and anchors), no_invented_facts=not invented, minimal_edit=bool(minimal),
        details=dict(segments_in_order=in_order, polarity_unchanged=neg_same, numbers_kept=nums_kept, final_punct_same=punct_same,
                     span_words_kept=anchors, invented_tokens=invented, tokens_added_per_edit=added,
                     edit_ratio=round(sum(added) / max(1, len(o_t)), 3)))


# ----------------------------------------------------------------------------- 8. full pipeline
def run(message, history, sender_context=None, meta=None, recipient="recipient", llm=None):
    sc, warnings = sender_context or {}, []
    meta = dict(sc.get("meta") or {}, **(meta or {}))
    ref_date = dt.date.fromisoformat(sc.get("reference_date")) if sc.get("reference_date") else dt.date.today()
    prof = build_profile(history, meta, recipient)
    cands, analyzer = detect(message, llm, warnings)
    cands = score_spans(cands, prof, sc, message)
    decision = aot_decide(cands)
    selected = goat_select(cands) if decision["action"] == "intervene" else []

    for c in selected:
        c["proposal"], c["evidence"], c["question"] = propose(c, message, sc, meta, ref_date)
    todo = [c for c in selected if c["proposal"]]
    llm_out = {}
    if llm is not None and llm.remote and todo:
        try:
            llm_out = llm.rewrite(message, [dict(id=i, type=c["type"], span=c["span"], proposal=c["proposal"], evidence=c["evidence"])
                                            for i, c in enumerate(todo)])
        except Exception as e:
            warnings.append(f"LLM rewrite failed, used templates: {type(e).__name__}: {e}")
    edits = []
    for i, c in enumerate(todo):
        cand = llm_out.get(i)
        if cand and edit_ok(c, cand, c["evidence"]):
            c["replacement"], c["source"] = cand, "llm"
        else:
            if cand:
                warnings.append(f"LLM replacement for '{c['span']}' rejected by constraint check; template used.")
            c["replacement"], c["source"] = c["proposal"], "template"
        if edit_ok(c, c["replacement"], c["evidence"]):
            edits.append(c)
        else:
            c["question"], c["replacement"] = f"Supplied context for '{c['span']}' does not keep the original wording; please confirm.", None
    adapted, segs = assemble(message, edits)
    cons = check_constraints(message, adapted, edits)
    if not all(cons[k] for k in ("intent_preserved", "no_invented_facts", "minimal_edit")):  # fall back to templates, then abstain
        for c in edits:
            c["replacement"], c["source"] = c["proposal"], "template"
        adapted, segs = assemble(message, edits)
        cons = check_constraints(message, adapted, edits)
        if not all(cons[k] for k in ("intent_preserved", "no_invented_facts", "minimal_edit")):
            warnings.append("Constraint check failed twice; returning the original message unchanged.")
            edits, adapted, segs = [], message, [{"text": message, "edited": False}]
            cons = check_constraints(message, adapted, [])
    edited_ids = {id(c) for c in edits}
    unresolved = [c for c in selected if id(c) not in edited_ids]
    resid = round(noisy_or([c["risk"] * CFG["residual_factor"] if id(c) in edited_ids else c["risk"] for c in cands if c["active"]]), 4)
    outcome = ("pass" if not selected else "abstained" if not edits and any(c.get("proposal") for c in selected) else
               "needs_sender_input" if not edits else "partial" if unresolved else "adapted")
    spans = []
    for c in cands:
        iv = None
        if c in selected:
            iv = dict(action=("replace" if id(c) in edited_ids else "ask_sender"), replacement=c.get("replacement"),
                      source=c.get("source"), evidence=c.get("evidence"),
                      question=c.get("question") if id(c) not in edited_ids else None)
        spans.append({k: c[k] for k in ("type", "kind", "span", "start", "end", "reason", "base", "grounding", "grounding_note",
                                         "clarif_rate", "risk", "active")} | dict(selected=c in selected, intervention=iv))
    return dict(
        recipient=recipient, original=message, adapted=adapted, adapted_segments=segs, outcome=outcome, decision=decision,
        goat=dict(selected=[c["span"] for c in selected], residual_risk=resid,
                  note="Greedy: add highest-risk spans until residual message risk < tau_low (an edited span keeps "
                       f"{int(CFG['residual_factor'] * 100)}% of its risk)."),
        spans=spans, constraints=cons, needs_sender_input=[dict(span=c["span"], question=c["question"]) for c in unresolved],
        profile=profile_summary(prof), analyzer=analyzer, warnings=warnings, reference_date=ref_date.isoformat())
