import json
import os
import tempfile
import unittest

from moe_residency.adapters import colibri, freetoken, llama_cpp
from moe_residency.profile import DomainProfile


class TestColibri(unittest.TestCase):
    def test_env_recipe_has_documented_keys(self):
        env = colibri.env_recipe("/models/x")
        for k in ("COLI_MODEL", "COLI_USAGE", "PIN", "PIN_GB", "PREFETCH",
                  "PILOT_REAL", "PILOT_EVICT_GUARD"):
            self.assertIn(k, env)

    def test_pipe_toggle(self):
        self.assertIn("PIPE", colibri.env_recipe("/m", pipe=True))
        self.assertNotIn("PIPE", colibri.env_recipe("/m", pipe=False))

    def test_half_life_decay(self):
        self.assertAlmostEqual(colibri.decay_for_half_life(10), 0.5 ** 0.1, places=6)
        self.assertEqual(colibri.decay_for_half_life(0), 1.0)

    def test_notes_mention_undisclosed_format(self):
        joined = " ".join(colibri.notes())
        self.assertIn(".coli_usage", joined)


class TestLlamaCpp(unittest.TestCase):
    def test_hotset_export(self):
        p = DomainProfile(32)
        for _ in range(4):
            p.note("code", [1, 2, 3])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "hot.json")
            llama_cpp.export_hotset(p, path)
            data = json.load(open(path, encoding="utf-8"))
        self.assertEqual(data["format"], "moe-residency/hotset/1")
        self.assertEqual(data["domains"]["code"][:3], [1, 2, 3])

    def test_cli_recipe(self):
        cmd = llama_cpp.cli_recipe(n_cpu_moe=20)
        self.assertIn("--n-cpu-moe", cmd)
        self.assertIn("--n-gpu-layers", cmd)


class TestFreeToken(unittest.TestCase):
    def test_hook_points_mention_semantic_anchor(self):
        joined = " ".join(freetoken.hook_points())
        self.assertIn("semantic anchor", joined.lower())

    def test_notes_say_do_not_touch_qstar(self):
        joined = " ".join(freetoken.integration_notes())
        self.assertIn("q*", joined)


if __name__ == "__main__":
    unittest.main()
