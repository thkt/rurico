"""Exercise the pinned official wrapper with synthetic logits, without model downloads."""
import itertools
import types
import unittest
from unittest.mock import Mock, patch

import torch
from sentence_transformers import CrossEncoder
from transformers import BatchEncoding

import reference


class RerankerCapture(unittest.TestCase):
    def setUp(self):
        self.logits = [-2.0, 3.0, 80.0]
        model = torch.nn.Module()
        model.register_parameter("weight", torch.nn.Parameter(torch.zeros(1)))
        model.config = types.SimpleNamespace(num_labels=1, _attn_implementation="eager",
                                            reference_compile=False)
        model.device = torch.device("cpu")
        model.forward = Mock(side_effect=itertools.cycle(
            types.SimpleNamespace(logits=torch.tensor([[x]])) for x in self.logits))
        self.wrapper = CrossEncoder.__new__(CrossEncoder)
        torch.nn.Module.__init__(self.wrapper)
        self.wrapper.model = model
        self.wrapper.activation_fn = torch.nn.Sigmoid()

        def tokenize(*args, **kwargs):
            values = {"input_ids":[1, 2], "attention_mask":[1, 1]}
            if kwargs.get("return_tensors") == "pt":
                return BatchEncoding({k:torch.tensor([v]) for k,v in values.items()})
            return values

        self.wrapper.tokenizer = Mock(side_effect=tokenize)
        self.inputs = {"reranker":[{"id":str(i), "query":"query", "text":f"text {i}", "repeat":1}
                                   for i in range(len(self.logits))]}

    def capture(self):
        # Only construction is replaced; CrossEncoder.predict and Torch sigmoid are real.
        with patch("sentence_transformers.CrossEncoder", return_value=self.wrapper):
            return reference.capture("reranker", "unused", {"schema":2, "batches":[]}, self.inputs)

    def test_one_forward_records_raw_logit_and_official_score_per_input(self):
        result = self.capture()
        self.assertEqual(self.wrapper.model.forward.call_count, len(self.logits))
        self.assertEqual(result["native"], [
            {"id":str(i), "value":[x, torch.nn.Sigmoid()(torch.tensor(x)).item()]}
            for i,x in enumerate(self.logits)])
        # Separate raw/encoded token observations plus the official predict tokenizer call.
        calls = self.wrapper.tokenizer.call_args_list
        self.assertEqual(len(calls), 3 * len(self.logits))
        for i,call in enumerate(calls[2::3]):
            self.assertEqual(call.args, ([("query", f"text {i}")],))
            self.assertEqual(call.kwargs, {"padding":True, "truncation":True, "return_tensors":"pt"})

    def test_forward_and_activation_failures_propagate_without_retry(self):
        for stage in ("forward", "activation"):
            with self.subTest(stage=stage):
                self.wrapper.model.forward.reset_mock()
                error = RuntimeError(f"{stage} failed")
                target = (patch.object(self.wrapper.model, "forward", side_effect=error)
                          if stage == "forward" else patch("torch.sigmoid", side_effect=error))
                with target:
                    with self.assertRaises(RuntimeError) as raised:
                        self.capture()
                    self.assertIs(raised.exception, error)
                    self.assertEqual(self.wrapper.model.forward.call_count, 1)


if __name__ == "__main__":
    unittest.main()
