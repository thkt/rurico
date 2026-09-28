"""Exercise the installed wrappers with tiny local models, without downloads."""
import itertools
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from sentence_transformers import CrossEncoder
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (ModernBertConfig, ModernBertModel,
                          ModernBertForSequenceClassification, PreTrainedTokenizerFast)
from transformers.modeling_outputs import SequenceClassifierOutput

import reference


def local_model(path, model_class):
    tokenizer = Tokenizer(models.WordLevel(
        {"[PAD]":0, "[UNK]":1, "query":2, "text":3, "0":4, "1":5, "2":6}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]",
                           model_max_length=8192).save_pretrained(path)
    config = ModernBertConfig(vocab_size=7, hidden_size=8, intermediate_size=16,
                             num_hidden_layers=1, num_attention_heads=2, num_labels=1,
                             pad_token_id=0, bos_token_id=1, eos_token_id=1,
                             cls_token_id=1, sep_token_id=1)
    model_class(config).save_pretrained(path)


class RerankerCapture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        local_model(self.directory.name, ModernBertForSequenceClassification)
        self.wrapper = CrossEncoder(self.directory.name, device="cpu", max_length=8192,
                                    local_files_only=True, model_kwargs={"dtype":torch.float32,
                                                                       "attn_implementation":"eager"})
        self.logits = [-2.0, 3.0, 80.0]
        self.wrapper.model.forward = Mock(side_effect=itertools.cycle(
            SequenceClassifierOutput(logits=torch.tensor([[x]])) for x in self.logits))
        self.inputs = {"reranker":[{"id":str(i), "query":"query", "text":f"text {i}", "repeat":1}
                                   for i in range(len(self.logits))]}

    def capture(self):
        # Construction has already used the installed wrapper and a tiny local checkpoint.
        # predict, preprocessing, the wrapper's modules and Torch sigmoid remain real.
        with patch("sentence_transformers.CrossEncoder", return_value=self.wrapper):
            return reference.capture("reranker", self.directory.name,
                                     {"schema":2, "batches":[]}, self.inputs)

    def test_one_forward_records_raw_logit_and_official_score_per_input(self):
        with patch.object(type(self.wrapper.tokenizer), "__call__", autospec=True,
                          side_effect=type(self.wrapper.tokenizer).__call__) as tokenize:
            result = self.capture()
        self.assertEqual(self.wrapper.model.forward.call_count, len(self.logits))
        self.assertEqual(result["native"], [
            {"id":str(i), "value":[x, torch.nn.Sigmoid()(torch.tensor(x)).item()]}
            for i,x in enumerate(self.logits)])
        # Separate raw/encoded observations plus predict's own preprocessing.
        self.assertEqual(tokenize.call_count, 3 * len(self.logits))
        for i, call in enumerate(tokenize.call_args_list[2::3]):
            self.assertEqual(call.args[1:], ([("query", f"text {i}")],))
            self.assertEqual(call.kwargs["padding"], True)
            self.assertEqual(call.kwargs["truncation"], "longest_first")
            self.assertEqual(call.kwargs["return_tensors"], "pt")
            expected_ids = [2, 3, 4+i]
            tokens = result["tokenization"][i]
            self.assertEqual(tokens["model_ids"], expected_ids)
            self.assertEqual(tokens["raw_ids"], expected_ids)
            self.assertEqual(tokens["model_mask"], [1, 1, 1])
            self.assertEqual(self.wrapper.model.forward.call_args_list[i].kwargs["input_ids"].tolist(),
                             [expected_ids])
        self.assertEqual(self.wrapper.max_seq_length, 8192)

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

    def test_compiled_module_is_rejected_before_observation(self):
        self.wrapper.model._compiled_call_impl = Mock()
        with self.assertRaisesRegex(ValueError, "compilation must remain disabled"):
            self.capture()
        self.wrapper.model.forward.assert_not_called()


class EmbeddingCapture(unittest.TestCase):
    def test_local_wrapper_loads_pooling_and_replays_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            local_model(path, ModernBertModel)
            # The fixed real model uses the legacy module/pooling serialization.
            (path/"modules.json").write_text(
                '[{"idx":0,"name":"0","path":"","type":"sentence_transformers.models.Transformer"},'
                '{"idx":1,"name":"1","path":"1_Pooling","type":"sentence_transformers.models.Pooling"}]')
            (path/"1_Pooling").mkdir()
            (path/"1_Pooling/config.json").write_text(
                '{"word_embedding_dimension":8,"pooling_mode_cls_token":false,'
                '"pooling_mode_mean_tokens":true,"include_prompt":true}')
            inputs = {"embedding":[{"id":"short", "prefix":"", "text":"query text", "repeat":1}]}
            observed = {"schema":2, "batches":[{"id":"short/exact", "keys":["short"],
                        "ids":[[2, 3]], "mask":[[1, 1]], "conditions":["exact", "bucket"]}]}
            result = reference.capture("embedding", path, observed, inputs)
        self.assertEqual(result["tokenization"][0]["model_ids"], [2, 3])
        native = result["native"][0]["value"]
        replay = result["batches"][0]["values"][0]
        self.assertEqual(len(native), 8)
        self.assertAlmostEqual(math.sqrt(sum(x*x for x in native)), 1.0, places=6)
        for left, right in zip(native, replay):
            self.assertAlmostEqual(left, right, places=6)
        self.assertEqual(len(result["batches"][0]["hidden_probes"][0]), 3)


if __name__ == "__main__":
    unittest.main()
