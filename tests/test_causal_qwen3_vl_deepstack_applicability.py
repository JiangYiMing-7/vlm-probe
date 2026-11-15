"""Qwen3-VL declares the residual layers that precede later DeepStack injections."""
import importlib.util
from importlib.metadata import version
import unittest
import warnings

import torch

from vlm_probing import Prober, TokenMargin
from vlm_probing.adapters.spec import ModelSpec
from vlm_probing.api.residual_applicability import ResidualInterventionWarning

HAS_V5 = (importlib.util.find_spec("transformers") is not None
          and version("transformers").split(".")[:2] == ["5", "3"])


def tiny_qwen3_vl(deepstack):
    import transformers as hf

    torch.manual_seed(0)
    text = dict(vocab_size=40, hidden_size=16, intermediate_size=24, num_hidden_layers=4,
                num_attention_heads=2, num_key_value_heads=1, pad_token_id=0, head_dim=8,
                rope_parameters=dict(rope_type="default", mrope_section=[1, 1, 2]))
    vision = dict(depth=3, hidden_size=16, intermediate_size=24, num_heads=2, patch_size=2,
                  temporal_patch_size=1, spatial_merge_size=2, out_hidden_size=16,
                  num_position_embeddings=16, deepstack_visual_indexes=deepstack)
    config = hf.Qwen3VLConfig(text_config=text, vision_config=vision, image_token_id=28,
                              video_token_id=30, vision_start_token_id=27, vision_end_token_id=29)
    config._attn_implementation = "eager"
    model = hf.Qwen3VLForConditionalGeneration(config).eval()
    inputs = {"input_ids": torch.tensor([[1, 27, 28, 28, 29, 3]]),
              "attention_mask": torch.ones(1, 6, dtype=torch.long),
              "image_grid_thw": torch.tensor([[1, 2, 4]]), "pixel_values": torch.randn(8, 12)}
    return model, inputs


class SpecTests(unittest.TestCase):
    def test_frozenset_declaration_is_validated(self):
        ModelSpec({0: "r0", 1: "r1"}, residual_bypass=frozenset({0})).validate({"r0": None, "r1": None})
        for value, error in ((frozenset({"0"}), TypeError), ({0}, TypeError), (frozenset({5}), ValueError)):
            with self.subTest(value=value), self.assertRaises(error):
                ModelSpec({0: "r0", 1: "r1"}, residual_bypass=value).validate({"r0": None, "r1": None})


@unittest.skipUnless(HAS_V5, "native Qwen3-VL requires Transformers 5.3.x")
class Qwen3VLDeepStackTests(unittest.TestCase):
    def test_declared_layers_are_exactly_those_that_fail_restoration(self):
        model, inputs = tiny_qwen3_vl([0, 1, 2])
        source = {**inputs, "pixel_values": -inputs["pixel_values"]}
        probe, metric = Prober(model), TokenMargin(4, 5)
        self.assertEqual(probe.spec.residual_bypass, frozenset({0, 1}))
        with torch.no_grad():
            total = (metric.score(probe._run(source).logits, probe.spec.layout(source))
                     - metric.score(probe._run(inputs).logits, probe.spec.layout(inputs))).item()
        with self.assertWarnsRegex(ResidualInterventionWarning, r"layers \[0, 1\]"):
            method = probe.causal.patch(layers=[0, 1, 2, 3], tokens="all")
        result = method.run(inputs, source=source, metric=metric)
        self.assertEqual(result.metadata["applicability"]["restoration_incomplete_layers"], [0, 1])
        gaps = [abs(e - total) for e in result.tensors["effect"][:, 0].tolist()]
        self.assertTrue(gaps[0] > 1e-4 and gaps[1] > 1e-4)
        self.assertLess(max(gaps[2:]), 1e-5)
        with self.assertWarns(ResidualInterventionWarning):
            probe.causal.attribute(layers=[1])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIsNone(probe.causal.patch(layers=[2, 3]).applicability)
        self.assertFalse(caught)

    def test_single_deepstack_level_declares_nothing(self):
        model, _ = tiny_qwen3_vl([0])
        probe = Prober(model)
        self.assertIs(probe.spec.residual_bypass, False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIsNone(probe.causal.patch(layers=[0, 1]).applicability)
        self.assertFalse(caught)


if __name__ == "__main__":
    unittest.main()
