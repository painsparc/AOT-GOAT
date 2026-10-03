"""Small controlled evaluation of the AOT-GOAT MVP.  Run:  python evaluate.py

IMPORTANT (read before quoting numbers): data/eval_set.json is a SMALL, SYNTHETIC set written by the project author.
Gold labels = "which risk types genuinely need an intervention for THIS recipient history". It is split into `dev`
(used while building/tuning the rules) and `holdout` (written afterwards, never used for tuning). The numbers measure
whether the pipeline behaves as designed on controlled cases; they are NOT evidence of real-world performance.
"""
import json
import os
import sys

import aotgoat
from aotgoat import CFG, TYPES, check_constraints

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_PATH = os.path.join(HERE, "data", "eval_set.json")


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=round(p, 3), recall=round(r, 3), f1=round(2 * p * r / (p + r) if p + r else 0.0, 3), tp=tp, fp=fp, fn=fn)


def _confusion(pairs):
    tp = sum(1 for g, p in pairs if g and p)
    fp = sum(1 for g, p in pairs if not g and p)
    fn = sum(1 for g, p in pairs if g and not p)
    tn = sum(1 for g, p in pairs if not g and not p)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, **{k: v for k, v in prf(tp, fp, fn).items() if k in ("precision", "recall", "f1")})


def _independent_minimality(out):
    """No text outside selected spans may change: unedited segments must equal the original minus the selected spans."""
    cut, pos = [], 0
    for s in sorted((s for s in out["spans"] if s["intervention"] and s["intervention"]["action"] == "replace"), key=lambda s: s["start"]):
        cut.append(out["original"][pos:s["start"]])
        pos = s["end"]
    cut.append(out["original"][pos:])
    kept = [g["text"] for g in out["adapted_segments"] if not g["edited"]]
    return "".join(cut) == "".join(kept)


def _score(cases, llm=None):
    rows, type_pairs = [], {t: [] for t in TYPES}
    for c in cases:
        out = aotgoat.run(c["message"], c["history"], c.get("sender_context"), c.get("meta"), c["id"], llm)
        gold = set(c["gold"])
        pred = {s["type"] for s in out["spans"] if s["risk"] >= CFG["tau_span"]}
        for t in TYPES:
            type_pairs[t].append((t in gold, t in pred))
        rows.append(dict(id=c["id"], gold=sorted(gold), pred=sorted(pred), action=out["decision"]["action"], outcome=out["outcome"],
                         message_risk=out["decision"]["message_risk"], out=out))
    all_pairs = [p for t in TYPES for p in type_pairs[t]]
    case_pairs = [(bool(r["gold"]), r["action"] == "intervene") for r in rows]
    edited = [r for r in rows if r["outcome"] in ("adapted", "partial")]
    cons = {k: sum(1 for r in edited if r["out"]["constraints"][k]) for k in ("intent_preserved", "no_invented_facts", "minimal_edit")}
    return dict(
        n_cases=len(rows),
        span_type_level=dict(note=f"one binary label per (case, risk type); predicted positive = a span of that type has risk >= {CFG['tau_span']}",
                             micro=_confusion(all_pairs), per_type={t: _confusion(type_pairs[t]) for t in TYPES}),
        decision_level=dict(note="gold = any risk type needs intervention for this recipient; predicted = AOT decision 'intervene'",
                            confusion=_confusion(case_pairs)),
        constraint_checks=dict(
            n_adapted_outputs=len(edited), **{f"{k}_pass": v for k, v in cons.items()},
            all_three_pass=sum(1 for r in edited if all(r["out"]["constraints"][k] for k in cons)),
            pass_through_unchanged=sum(1 for r in rows if r["action"] == "pass" and r["out"]["adapted"] == r["out"]["original"]),
            n_pass_decisions=sum(1 for r in rows if r["action"] == "pass"),
            independent_no_unselected_edits=sum(1 for r in rows if _independent_minimality(r["out"])),
            n_with_independent_check=len(rows)),
        errors=[dict(id=r["id"], gold=r["gold"], pred=r["pred"]) for r in rows if r["gold"] != r["pred"]],
        _rows=rows)


def checker_self_test(rows):
    """Inject deliberate violations into real edits; the constraint checker must flag every one of them."""
    res = {"invented_fact": [0, 0], "dropped_negation": [0, 0], "over_long_edit": [0, 0], "meaning_changing_span_loss": [0, 0]}
    for r in rows:
        out = r["out"]
        sel = [s for s in out["spans"] if s["intervention"] and s["intervention"]["action"] == "replace"]
        if not sel:
            continue
        base = [dict(type=s["type"], span=s["span"], start=s["start"], end=s["end"], replacement=s["intervention"]["replacement"],
                     evidence=s["intervention"]["evidence"]) for s in sel]  # the real evidence the pipeline allowed
        variants = {
            "invented_fact": lambda e: dict(e, replacement=e["replacement"] + " Priya Sharma from Acme"),
            "dropped_negation": lambda e: dict(e, replacement=e["replacement"] + " not"),
            "over_long_edit": lambda e: dict(e, replacement=e["replacement"] + " " + " ".join(["x%d" % i for i in range(20)])),
            "meaning_changing_span_loss": lambda e: dict(e, replacement="something else entirely"),
        }
        for name, fn in variants.items():
            bad = [fn(base[0])] + base[1:]
            adapted, _ = aotgoat.assemble(out["original"], bad)
            c = check_constraints(out["original"], adapted, bad)
            flagged = not (c["intent_preserved"] and c["no_invented_facts"] and c["minimal_edit"])
            res[name][0] += int(flagged)
            res[name][1] += 1
    return {k: dict(detected=v[0], injected=v[1]) for k, v in res.items()}


def evaluate(path=EVAL_PATH, llm=None):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cases = data["cases"]
    report = dict(disclaimer=data.get("disclaimer", ""), analyzer=("llm+rules" if llm is not None and llm.remote else "offline-rules (deterministic)"),
                  config={k: CFG[k] for k in CFG}, splits={})
    rows_all = []
    for name in ("dev", "holdout", "all"):
        subset = cases if name == "all" else [c for c in cases if c.get("split") == name]
        if not subset:
            continue
        s = _score(subset, llm)
        rows_all = s["_rows"] if name == "all" else rows_all
        s.pop("_rows") if name != "all" else None
        report["splits"][name] = s
    report["checker_self_test"] = checker_self_test(report["splits"]["all"].pop("_rows"))
    return report


def _print(rep):
    print(f"\nAOT-GOAT evaluation  |  analyzer: {rep['analyzer']}\n{rep['disclaimer']}\n")
    for name, s in rep["splits"].items():
        m = s["span_type_level"]["micro"]
        d = s["decision_level"]["confusion"]
        print(f"=== split: {name} ({s['n_cases']} cases) ===")
        print(f"Type-level (micro)   P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}   TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
        for t, c in s["span_type_level"]["per_type"].items():
            print(f"   {t:12} P={c['precision']:.3f} R={c['recall']:.3f} F1={c['f1']:.3f}  (TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']})")
        print("Decision-level confusion (rows = gold, cols = predicted)")
        print(f"                 pred pass   pred intervene\n   gold pass      {d['tn']:>6}      {d['fp']:>8}\n   gold intervene {d['fn']:>6}      {d['tp']:>8}")
        print(f"   P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f}")
        k = s["constraint_checks"]
        print(f"Constraints on {k['n_adapted_outputs']} adapted outputs: intent {k['intent_preserved_pass']}, no-invented-facts "
              f"{k['no_invented_facts_pass']}, minimal {k['minimal_edit_pass']}, all-three {k['all_three_pass']}; pass-through unchanged "
              f"{k['pass_through_unchanged']}/{k['n_pass_decisions']}; no unselected text edited {k['independent_no_unselected_edits']}/{k['n_with_independent_check']}")
        if s["errors"]:
            print("Errors (id: gold -> predicted):")
            for e in s["errors"]:
                print(f"   {e['id']}: {e['gold']} -> {e['pred']}")
        print()
    print("Checker self-test (injected violations detected / injected):")
    for k, v in rep["checker_self_test"].items():
        print(f"   {k:28} {v['detected']}/{v['injected']}")


if __name__ == "__main__":
    from llm import LLM
    rep = evaluate(llm=LLM() if "--llm" in sys.argv else None)
    _print(rep)
    with open(os.path.join(HERE, "eval_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("\nSaved eval_report.json")
