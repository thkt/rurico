"""Small counterexamples for the investigation's evaluator, no model required."""
import copy
import math
import unittest

from compare import vector_diff, reranker_diff, ranking_diff, validate_batch, compare
from search_quality import metrics_delta


def reranker_records(boundary=False):
    cases = [{"id":k,"query":"q","text":"d","repeat":1} for k in ("a","b","c","d","e")]
    cases[-1]["query"] = "another query"
    values = [[0.,.5],[1.,.7310585786],[2.,.880797078],[3.,.952574127],[4.,.98201379]]
    tokens = [{"id":x["id"],"raw_ids":[1,5,2],"raw_mask":[1,1,1],
               "model_ids":[1,5,2],"model_mask":[1,1,1]} for x in cases]
    if boundary:
        tokens[-1].update(raw_ids=[1]*128, raw_mask=[1]*128,
                          model_ids=[1]*128, model_mask=[1]*128)
    batches = []
    for c,v,t in zip(cases,values,tokens):
        ids, mask = t["model_ids"], t["model_mask"]
        for mode,length in (("exact",len(ids)),("bucket",128)):
            batches.append(dict(id=c["id"]+"/text/"+mode,keys=[c["id"]+"/text"],
                ids=[ids+[0]*(length-len(ids))],mask=[mask+[0]*(length-len(ids))],values=[v],hidden_probes=[]))
    batches.append(dict(id="mixed/bucket",keys=[c["id"]+"/text" for c in cases],
                        ids=[t["model_ids"]+[0]*(128-len(t["model_ids"])) for t in tokens],
                        mask=[t["model_mask"]+[0]*(128-len(t["model_mask"])) for t in tokens],
                        values=values,hidden_probes=[]))
    actual = dict(schema=1,producer="rurico",kind="reranker",tokenization=tokens,batches=batches,
                  public=dict(singleton=[dict(id=c["id"],score=v[1]) for c,v in zip(cases,values)],
                              mixed_scores=[v[1] for v in values],ranking=[dict(index=i,score=values[i][1]) for i in (3,2,1,0)]))
    reference = dict(schema=1,producer="transformers",kind="reranker",tokenization=copy.deepcopy(tokens),
                     batches=copy.deepcopy(batches),native=[dict(id=c["id"],value=v) for c,v in zip(cases,values)])
    return actual,reference,{"reranker":cases}


class Diagnostics(unittest.TestCase):
    def test_shared_observation_preserves_drift_and_rejects_false_sharing(self):
        actual,reference,inputs = reranker_records(boundary=True)
        legacy = compare(actual,reference,inputs)
        for record in (actual,reference):
            record["schema"] = 2
            record["batches"] = [b for b in record["batches"] if b["id"] != "e/text/bucket"]
            for batch in record["batches"]:
                batch["conditions"] = (["exact","bucket"] if batch["id"] == "e/text/exact"
                                       else [batch["id"].rsplit("/",1)[1]])
        result = compare(actual,reference,inputs)
        self.assertEqual(result["same_tokens"], [b for b in legacy["same_tokens"] if b["batch"] != "e/text/bucket"])
        self.assertEqual(result["ranking"], legacy["ranking"])
        self.assertEqual(result["wrapper_and_chunks"], legacy["wrapper_and_chunks"])
        self.assertEqual(result["padding"][-1]["observations"], {"exact":"e/text/exact","bucket":"e/text/exact"})
        self.assertTrue(result["padding"][-1]["same_observation"])
        self.assertFalse(result["padding"][0]["same_observation"])
        bad = copy.deepcopy(actual)
        bad["batches"][-2]["values"][0][0] += .001
        self.assertFalse(compare(bad,reference,inputs)["same_token_numerics_within_criteria"])
        for target in (actual,reference):
            original = target["batches"][0]["conditions"]
            target["batches"][0]["conditions"] = ["exact","bucket"]
            with self.assertRaisesRegex(ValueError, "batch conditions"):
                compare(actual,reference,inputs)
            target["batches"][0]["conditions"] = original
        actual["batches"].pop(-2)
        with self.assertRaisesRegex(ValueError,"records"):
            compare(actual,reference,inputs)

    def test_embedding_requires_cosine_and_absolute_difference(self):
        self.assertTrue(vector_diff([1., 0.], [1., 0.])["within_criteria"])
        # Equal direction alone hides magnitude/normalization drift.
        self.assertFalse(vector_diff([1., 0.], [1.00002, 0.])["within_criteria"])
        # Small coordinates alone can hide a rotation in a small-norm vector.
        self.assertFalse(vector_diff([1e-6, 0.], [0., 1e-6])["within_criteria"])
        for bad in ([math.nan, 0.], [math.inf, 0.], [0., 0.], [1.]):
            with self.assertRaises(ValueError):
                vector_diff([1., 0.], bad)

    def test_saturated_score_does_not_hide_logit_drift(self):
        self.assertTrue(reranker_diff([30., 1.], [30., 1.])["within_criteria"])
        self.assertFalse(reranker_diff([30., 1.], [30.001, 1.])["within_criteria"])
        self.assertFalse(reranker_diff([0., .5], [0., .50003])["within_criteria"])

    def test_ranking_keeps_near_reversals_and_stable_actual_order(self):
        result = ranking_diff(["a", "b", "c"], [1., 1.0001, 2.], [2., 1., 0.])
        self.assertEqual(result["reference_order"], ["c", "b", "a"])
        self.assertEqual(result["actual_order"], ["a", "b", "c"])
        self.assertEqual(len(result["near_pairs"]), 1)
        self.assertTrue(result["near_pairs"][0]["reversed"])
        self.assertEqual(len(result["discordant_pairs"]), 2)
        tied = ranking_diff(["a", "b"], [0., 1.], [0., 0.])
        self.assertFalse(tied["within_criteria"])

    def test_missing_row_mask_and_shape_fail_closed(self):
        good = {"id":"mixed", "keys":["a", "b"], "ids":[[1,2],[1,3]],
                "mask":[[1,1],[1,0]], "values":[[0.,.5],[1.,.731]], "hidden_probes":[]}
        validate_batch(good, "reranker")
        for field, bad in (("values", [[0.,.5]]), ("mask", [[1,1],[0,0]]),
                           ("ids", [[1,2],[1]]), ("keys", ["a","a"])):
            with self.assertRaises(ValueError, msg=field):
                validate_batch({**good,field:bad}, "reranker")

    def test_complete_comparison_detects_missing_record_native_nan_and_returned_rank(self):
        actual,reference,inputs = reranker_records()
        result = compare(actual,reference,inputs)
        self.assertTrue(result["same_token_numerics_within_criteria"])
        self.assertTrue(result["ranking"][-1]["within_criteria"])
        # Structural validation alone need not inspect numbers, but compare must
        # reject invalid values from either producer, including mixed-only rows.
        for producer in ("rurico", "transformers"):
            for value in (math.nan, math.inf, -math.inf, True, "0"):
                with self.subTest(producer=producer, batch_value=value):
                    a, r = copy.deepcopy((actual, reference))
                    target = a if producer == "rurico" else r
                    target["batches"][-1]["values"][-1] = [value, .5]
                    with self.assertRaisesRegex(ValueError, "non-finite"):
                        compare(a,r,inputs)
        bad = copy.deepcopy(actual)
        bad["public"]["singleton"][-1]["score"] = math.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compare(bad,reference,inputs)
        # Native shape must be checked before indexing its score. Batch validation
        # and the diff helpers alone cannot catch a misplaced check in compare().
        for value, error in (([], "expected logit and score"),
                             ([0.], "expected logit and score"),
                             ([0., .5, 1.], "expected logit and score"),
                             ([math.nan, .5], "non-finite"),
                             ([0., math.inf], "non-finite"),
                             ([0., 1.1], "score outside")):
            with self.subTest(native=value):
                bad = copy.deepcopy(reference)
                bad["native"][-1]["value"] = value
                with self.assertRaisesRegex(ValueError,error):
                    compare(actual,bad,inputs)
        bad = copy.deepcopy(reference)
        bad["batches"].pop()
        with self.assertRaisesRegex(ValueError,"records"):
            compare(actual,bad,inputs)
        actual["public"]["ranking"].reverse()
        self.assertFalse(compare(actual,reference,inputs)["ranking"][-1]["within_criteria"])

    def test_search_delta_keeps_ci_and_refuses_different_pipeline(self):
        baseline = dict(schema_version="1.3",kind="forward",fixture_hash="fixed",aggregation="identity",
                        merge_config={},normalization={})
        baseline["global"] = [dict(name="recall@10",k=10,point_estimate=.7,ci_lower=.6,ci_upper=.8)]
        measured = {**baseline,"global":[dict(name="recall@10",k=10,point_estimate=.8,ci_lower=.7,ci_upper=.9)]}
        result = metrics_delta(baseline,measured)[0]
        self.assertAlmostEqual(result["delta"],.1)
        self.assertEqual(result["measured"]["ci_upper"],.9)
        with self.assertRaisesRegex(ValueError,"condition differs"):
            metrics_delta(baseline,{**measured,"normalization":{"nfkc":False}})


if __name__ == "__main__":
    unittest.main()
