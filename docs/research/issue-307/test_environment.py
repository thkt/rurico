"""Reject drift between the one requirements list and the installed runtime."""
import types
import unittest
from unittest.mock import patch

import reference


class ReferenceEnvironment(unittest.TestCase):
    def check(self, pins, installed, *, isolated=True, python="3.12.13"):
        distributions = [types.SimpleNamespace(metadata={"Name":name}, version=version)
                         for name, version in installed.items()]
        with patch.object(reference.Path, "read_text", return_value=pins), \
             patch.object(reference.importlib.metadata, "distributions", return_value=distributions), \
             patch.object(reference.platform, "python_version", return_value=python), \
             patch.object(reference.sys, "prefix", "venv" if isolated else "base"), \
             patch.object(reference.sys, "base_prefix", "base"):
            return reference.environment()

    def setUp(self):
        # Deliberately different versions from the real requirements: a second
        # hard-coded version table must not reject a consistently updated set.
        self.pins = "# test pins\ntorch==10.0.0\ntransformers==11.0.0\nsentence-transformers==12.0.0\n"
        self.installed = {"torch":"10.0.0", "transformers":"11.0.0",
                          "Sentence_Transformers":"12.0.0", "pip":"26.2.1"}

    def test_one_manifest_accepts_normalized_names_but_rejects_runtime_drift(self):
        self.assertEqual(self.check(self.pins, self.installed)["sentence-transformers"], "12.0.0")
        self.check(self.pins.replace("torch==10.0.0", "torch==10.0.0.post1"),
                   dict(self.installed, torch="10.0.0.post1"))
        for installed in (dict(self.installed, torch="9.0.0"),
                          {k:v for k,v in self.installed.items() if k != "torch"},
                          dict(self.installed, unexpected_transitive="1.0.0")):
            with self.subTest(installed=installed), self.assertRaises(ValueError):
                self.check(self.pins, installed)

    def test_malformed_or_duplicate_pins_cannot_hide_from_validation(self):
        for pins in (self.pins+"Torch==10.0.0\n", self.pins.replace("torch==10.0.0", "torch>=10")):
            with self.subTest(pins=pins), self.assertRaises(ValueError):
                self.check(pins, self.installed)

        # Keep versions and dependency sets consistent so unrelated drift
        # checks cannot mask a missing stability or required-library guard.
        with self.assertRaisesRegex(ValueError, "^expected a stable dependency pin:"):
            self.check(self.pins.replace("torch==10.0.0", "torch==10.0.1rc1"),
                       dict(self.installed, torch="10.0.1rc1"))
        with self.assertRaisesRegex(ValueError, "^reference libraries must be pinned$"):
            self.check("# no runtime\n", {"pip":self.installed["pip"]})

    def test_python_version_and_isolation_remain_required(self):
        for options in ({"isolated":False}, {"python":"3.12.12"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.check(self.pins, self.installed, **options)


if __name__ == "__main__":
    unittest.main()
