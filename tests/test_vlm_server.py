import base64
import json
import unittest
import urllib.request

import _paths  # noqa: F401
import openai_server as vs
from g1voice.vlm import VLMClient, build_messages
from helpers import start_mock_vlm


class TestVLMServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv, cls.url = start_mock_vlm("one two three four")

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_streaming_text(self):
        c = VLMClient(self.url, "mock")
        out = "".join(c.stream(build_messages("sys", [], "hi", None)))
        self.assertEqual(out, "one two three four")

    def test_streaming_with_image(self):
        c = VLMClient(self.url, "mock")
        img = base64.b64encode(b"\xff\xd8fakejpeg").decode()
        out = "".join(c.stream(build_messages("sys", [], "what is this", img)))
        self.assertIn("image of 10 bytes", out)

    def test_non_streaming_and_timing(self):
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": False}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=5).read())
        self.assertEqual(d["choices"][0]["message"]["content"], "one two three four")
        self.assertEqual(d["x_timing"]["tokens"], 4)

    def test_bad_request(self):
        body = json.dumps({"messages": [{"role": "assistant", "content": "x"}]}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(ctx.exception.code, 400)

    def test_split_messages(self):
        img = base64.b64encode(b"abc").decode()
        msgs = build_messages("S", [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}],
                              "q2", img)
        system, hist, text, image = vs.split_messages(msgs)
        self.assertEqual((system, hist, text, image), ("S", [("user", "q1"), ("assistant", "a1")], "q2", b"abc"))


if __name__ == "__main__":
    unittest.main()
