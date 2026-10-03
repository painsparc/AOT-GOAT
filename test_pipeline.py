"""End-to-end tests.  Run:  python test_pipeline.py     (no network or API key needed; LLM calls are mocked)"""
import datetime as dt
import json
import os
import unittest
from unittest import mock

os.environ["LLM_PROVIDER"] = "offline"  # never let a local .env trigger real network calls in tests

import aotgoat  # noqa: E402
import app as appmod  # noqa: E402
import evaluate  # noqa: E402
import llm as llm_mod  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "data", "sample_data.json"), encoding="utf-8") as _f:
    SAMPLES = json.load(_f)["scenarios"]
S1 = SAMPLES[0]


def run_s1(i):
    r = S1["recipients"][i]
    return aotgoat.run(S1["message"], r["history"], S1["sender_context"], r.get("meta"), r["id"])


class TestProfile(unittest.TestCase):
    def test_vikram_profile(self):
        r = S1["recipients"][1]
        p = aotgoat.build_profile(r["history"], dict(S1["sender_context"]["meta"], **r["meta"]))
        self.assertGreaterEqual(p["clarification_counts"]["temporal"], 1)
        self.assertEqual(p["tz_gap_hours"], 10.5)
        self.assertEqual(p["n_recipient_messages"], 2)

    def test_asha_profile_terms(self):
        p = aotgoat.build_profile(S1["recipients"][0]["history"])
        self.assertIn("sla", [a.lower() for a in p["sender_terms"] + p["recipient_terms"]])
        self.assertTrue(p["depth"] > 0.5)

    def test_history_formats(self):
        a = aotgoat.norm_history(["S: hi", "R: hello"])
        b = aotgoat.norm_history([{"sender": "S", "text": "hi"}, {"sender": "R", "text": "hello"}])
        self.assertEqual(a, b)


class TestSameMessageDifferentOutputs(unittest.TestCase):
    def test_three_recipients_three_behaviours(self):
        asha, vikram, meera = run_s1(0), run_s1(1), run_s1(2)
        self.assertEqual(asha["outcome"], "pass")
        self.assertEqual(asha["adapted"], S1["message"])
        self.assertEqual(vikram["outcome"], "adapted")
        self.assertIn("Tue 6 Oct 2026", vikram["adapted"])
        self.assertIn("service-level agreement", vikram["adapted"])
        self.assertIn("the Q3 budget report", vikram["adapted"])
        self.assertEqual(meera["outcome"], "adapted")
        self.assertIn("service-level agreement", meera["adapted"])
        self.assertNotIn("Tue 6 Oct 2026", meera["adapted"])  # Meera already shares the EOD-tomorrow convention
        self.assertEqual(len({asha["adapted"], vikram["adapted"], meera["adapted"]}), 3)

    def test_all_five_risk_types_detected(self):
        types = {s["type"] for s in run_s1(1)["spans"]}
        self.assertEqual(types, set(aotgoat.TYPES))

    def test_output_has_everything_the_demo_needs(self):
        out = run_s1(1)
        for k in ("original", "adapted", "decision", "spans", "goat", "constraints", "profile"):
            self.assertIn(k, out)
        for s in out["spans"]:
            for k in ("span", "type", "reason", "base", "grounding", "risk", "start", "end"):
                self.assertIn(k, s)
            self.assertEqual(out["original"][s["start"]:s["end"]], s["span"])

    def test_deterministic(self):
        self.assertEqual(json.dumps(run_s1(1), sort_keys=True), json.dumps(run_s1(1), sort_keys=True))

    def test_goat_is_minimal(self):
        out = run_s1(2)  # Meera: only the ungrounded spans are edited
        edited = {s["span"] for s in out["spans"] if s["selected"]}
        self.assertTrue(edited <= {"it", "everyone", "SLA"})
        self.assertLess(out["goat"]["residual_risk"], aotgoat.CFG["tau_low"] + 0.2)


class TestHonestyAboutMissingFacts(unittest.TestCase):
    def test_vague_time_asks_sender_instead_of_inventing(self):
        out = aotgoat.run("Share your feedback soon.", ["S: Hi", "R: Hello"], {"reference_date": "2026-10-05"})
        self.assertEqual(out["outcome"], "needs_sender_input")
        self.assertEqual(out["adapted"], out["original"])
        self.assertEqual(out["needs_sender_input"][0]["span"], "soon")

    def test_unknown_pronoun_referent_asks_sender(self):
        out = aotgoat.run("He said he'd call them back.", ["S: Hi", "R: Hello"])
        self.assertEqual(out["adapted"], out["original"])
        self.assertTrue(out["needs_sender_input"])

    def test_negation_and_intent_survive_adaptation(self):
        out = aotgoat.run("Do not ship it until the review is done.", ["S: Hi", "R: Hello"],
                          {"referents": {"it": "the v2 build"}, "reference_date": "2026-10-05"})
        self.assertIn("Do not ship the v2 build", out["adapted"])
        self.assertTrue(out["constraints"]["intent_preserved"])
        self.assertTrue(out["constraints"]["no_invented_facts"])


class TestDates(unittest.TestCase):
    ref = dt.date(2026, 10, 5)  # a Monday

    def test_resolution(self):
        r = lambda e: aotgoat.resolve_date(e, self.ref)
        self.assertEqual(r("tomorrow"), "Tue 6 Oct 2026")
        self.assertEqual(r("friday"), "Fri 9 Oct 2026")
        self.assertIsNone(r("monday"))  # same weekday as today: ambiguous, so the sender is asked
        self.assertEqual(r("next monday"), "Mon 12 Oct 2026")
        self.assertEqual(r("next week"), "week of Mon 12 Oct 2026")
        self.assertEqual(r("in 3 days"), "Thu 8 Oct 2026")
        self.assertEqual(r("end of month"), "Sat 31 Oct 2026")
        self.assertIsNone(r("soon"))


class TestConstraintChecker(unittest.TestCase):
    orig = "Please send it to everyone. Do not delay."

    def _edit(self, rep, ev):
        return [dict(type="coreference", span="it", start=12, end=14, replacement=rep, evidence=ev)]

    def _check(self, rep, ev):
        e = self._edit(rep, ev)
        adapted, _ = aotgoat.assemble(self.orig, e)
        return aotgoat.check_constraints(self.orig, adapted, e)

    def test_valid_edit_passes(self):
        c = self._check("the Q3 report", ["the Q3 report"])
        self.assertTrue(c["intent_preserved"] and c["no_invented_facts"] and c["minimal_edit"])

    def test_invented_fact_is_flagged(self):
        c = self._check("the Q3 report from Priya", ["the Q3 report"])
        self.assertFalse(c["no_invented_facts"])
        self.assertIn("priya", c["details"]["invented_tokens"])

    def test_dropped_negation_is_flagged(self):
        e = self._edit("the Q3 report", ["the Q3 report"])
        self.assertFalse(aotgoat.check_constraints(self.orig, "Please send the Q3 report to everyone. Do delay.", e)["intent_preserved"])

    def test_over_long_edit_is_flagged(self):
        long = "the Q3 report " + " ".join(f"w{i}" for i in range(20))
        self.assertFalse(self._check(long, [long])["minimal_edit"])


class TestLLMPathMocked(unittest.TestCase):
    """Remote LLM behaviour with HTTP mocked: validation, rejection of bad generations, graceful fallback."""

    def _resp(self, text):
        m = mock.Mock()
        m.raise_for_status = lambda: None
        m.json = lambda: {"content": [{"type": "text", "text": text}]}
        return m

    def _fake(self, rewrite_text):
        def post(url, **kw):
            prompt = kw["json"]["messages"][0]["content"]
            if "Find spans" in prompt:
                return self._resp(json.dumps({"risks": [
                    {"type": "implicit", "span": "by the book", "severity": 0.7, "reason": "idiom"},
                    {"type": "scope", "span": "Zorblax", "severity": 0.9, "reason": "hallucinated"}]}))
            return self._resp(rewrite_text)
        return post

    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-key"})
        self.env.start()
        self.llm = llm_mod.LLM()
        self.sc = {"referents": {"it": "the invoice"}, "reference_date": "2026-10-05"}

    def tearDown(self):
        self.env.stop()

    def test_valid_llm_output_is_used_and_hallucinated_span_dropped(self):
        with mock.patch.object(llm_mod.requests, "post", self._fake('{"replacements": {"0": "the invoice"}}')):
            out = aotgoat.run("Please send it by the book.", ["S: Hi", "R: Hello"], self.sc, None, "r", self.llm)
        self.assertEqual(out["analyzer"], "llm(anthropic)+rules")
        self.assertIn("by the book", [s["span"] for s in out["spans"]])
        self.assertTrue(any("Zorblax" in w for w in out["warnings"]))
        self.assertIn("the invoice", out["adapted"])
        self.assertTrue(any(s["intervention"] and s["intervention"]["source"] == "llm" for s in out["spans"]))

    def test_llm_that_invents_facts_is_rejected(self):
        with mock.patch.object(llm_mod.requests, "post", self._fake('{"replacements": {"0": "the invoice from Priya Sharma"}}')):
            out = aotgoat.run("Please send it by the book.", ["S: Hi", "R: Hello"], self.sc, None, "r", self.llm)
        self.assertNotIn("Priya", out["adapted"])
        self.assertIn("the invoice", out["adapted"])
        self.assertTrue(any("rejected" in w for w in out["warnings"]))

    def test_network_failure_falls_back_to_offline_rules(self):
        with mock.patch.object(llm_mod.requests, "post", side_effect=llm_mod.requests.ConnectionError("down")):
            out = aotgoat.run("Please send it today.", ["S: Hi", "R: Hello"], self.sc, None, "r", self.llm)
        self.assertEqual(out["analyzer"], "offline-rules")
        self.assertTrue(any("LLM analysis failed" in w for w in out["warnings"]))
        self.assertIn("the invoice", out["adapted"])


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.c = appmod.app.test_client()

    def test_health_and_index(self):
        self.assertEqual(self.c.get("/api/health").get_json()["status"], "ok")
        r = self.c.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"AOT-GOAT", r.data)

    def test_samples(self):
        self.assertGreaterEqual(len(self.c.get("/api/samples").get_json()["scenarios"]), 2)

    def test_analyze(self):
        r = self.c.post("/api/analyze", json=dict(message=S1["message"], history=S1["recipients"][1]["history"],
                                                  sender_context=S1["sender_context"], meta=S1["recipients"][1]["meta"]))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["outcome"], "adapted")

    def test_compare(self):
        r = self.c.post("/api/compare", json=dict(message=S1["message"], sender_context=S1["sender_context"],
                                                  recipients=S1["recipients"]))
        res = r.get_json()["results"]
        self.assertEqual([x["outcome"] for x in res], ["pass", "adapted", "adapted"])

    def test_validation_errors(self):
        self.assertEqual(self.c.post("/api/analyze", json={"message": ""}).status_code, 400)
        self.assertEqual(self.c.post("/api/analyze", data="not json").status_code, 400)
        self.assertEqual(self.c.post("/api/compare", json={"message": "hi", "recipients": []}).status_code, 400)
        self.assertEqual(self.c.post("/api/analyze", json={"message": "hi", "history": "nope"}).status_code, 400)

    def test_evaluate_endpoint(self):
        j = self.c.get("/api/evaluate").get_json()
        self.assertIn("holdout", j["splits"])
        self.assertIn("disclaimer", j)


class TestEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rep = evaluate.evaluate()

    def test_metrics_are_well_formed(self):
        for name, s in self.rep["splits"].items():
            m = s["span_type_level"]["micro"]
            for k in ("precision", "recall", "f1"):
                self.assertTrue(0.0 <= m[k] <= 1.0, (name, k))
            d = s["decision_level"]["confusion"]
            self.assertEqual(d["tp"] + d["fp"] + d["fn"] + d["tn"], s["n_cases"])

    def test_constraints_hold_on_every_adapted_output(self):
        k = self.rep["splits"]["all"]["constraint_checks"]
        self.assertGreater(k["n_adapted_outputs"], 0)
        self.assertEqual(k["all_three_pass"], k["n_adapted_outputs"])
        self.assertEqual(k["pass_through_unchanged"], k["n_pass_decisions"])
        self.assertEqual(k["independent_no_unselected_edits"], k["n_with_independent_check"])

    def test_checker_catches_injected_violations(self):
        for name, v in self.rep["checker_self_test"].items():
            self.assertEqual(v["detected"], v["injected"], name)
            self.assertGreater(v["injected"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
