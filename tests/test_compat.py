"""Files that run on the Jetson host (system Python 3.8) or in the nano_llm r35 image
(Python 3.8) must parse as Python 3.8."""
import ast
import os
import unittest

import _paths

PY38 = ["services/vlm/openai_server.py", "scripts/capture_g1_mic.py", "scripts/record_utterances.py",
        "scripts/g1_operator.py", "scripts/bench/report.py", "scripts/bench/mem_report.py",
        "scripts/bench/_common.py"]


class TestPy38(unittest.TestCase):
    def test_syntax(self):
        for rel in PY38:
            path = os.path.join(_paths.ROOT, rel)
            if not os.path.exists(path):
                continue
            with self.subTest(rel):
                ast.parse(open(path).read(), filename=rel, feature_version=(3, 8))

    def test_no_stdlib_shadowing(self):
        import sys
        names = set(getattr(sys, "stdlib_module_names", ()))
        for d in ("scripts", "scripts/bench", "scripts/offline"):
            for fn in os.listdir(os.path.join(_paths.ROOT, d)):
                if fn.endswith(".py"):
                    self.assertNotIn(fn[:-3], names, "%s/%s shadows a stdlib module" % (d, fn))


if __name__ == "__main__":
    unittest.main()
