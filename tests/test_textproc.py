import unittest

import _paths  # noqa: F401
from g1voice.textproc import (CommandGrammar, GestureTagFilter, SentenceChunker, clean_for_speech,
                              has_wake_word, needs_vision)


def stream(chunker, filt, tokens):
    out = []
    for t in tokens:
        out += chunker.feed(filt.feed(t))
    out += chunker.feed(filt.flush())
    out += chunker.flush()
    return out


class TestChunker(unittest.TestCase):
    def test_first_chunk_three_words_then_eight(self):
        c = SentenceChunker(3, 8)
        text = "one two three four five six seven eight nine ten eleven twelve "
        out = []
        for w in text.split(" "):
            out += c.feed(w + " ")
        out += c.flush()
        self.assertEqual(out[0], "one two three")
        self.assertEqual(out[1], "four five six seven eight nine ten eleven")
        self.assertEqual(out[2], "twelve")

    def test_punctuation_flushes_early(self):
        c = SentenceChunker(3, 8)
        self.assertEqual(c.feed("Hi! "), ["Hi!"])
        self.assertEqual(c.feed("Nice to meet you. "), ["Nice to meet you."])

    def test_no_split_inside_word(self):
        c = SentenceChunker(3, 8)
        self.assertEqual(c.feed("hello wor"), [])
        self.assertEqual(c.feed("ld again"), [])
        self.assertEqual(c.feed(" "), ["hello world again"])

    def test_markdown_stripped(self):
        self.assertEqual(clean_for_speech("**Hello**   #world"), "Hello world")


class TestGestureFilter(unittest.TestCase):
    def test_tag_split_across_tokens(self):
        got = []
        f = GestureTagFilter({"wave"}, on_gesture=got.append)
        out = stream(SentenceChunker(3, 8), f, ["Hello", " there", "! [ges", "ture:wa", "ve] Bye."])
        self.assertEqual(got, ["wave"])
        self.assertEqual(" ".join(out), "Hello there! Bye.")

    def test_unknown_tag_is_stripped_and_reported(self):
        got, rej = [], []
        f = GestureTagFilter({"wave"}, on_gesture=got.append, on_rejected=rej.append)
        out = stream(SentenceChunker(3, 8), f, ["Okay [gesture:backflip] done."])
        self.assertEqual(got, [])
        self.assertEqual(rej, ["backflip"])
        self.assertNotIn("backflip", " ".join(out))

    def test_plain_brackets_pass_through(self):
        f = GestureTagFilter({"wave"})
        self.assertEqual(f.feed("see [note] here") + f.flush(), "see [note] here")

    def test_unclosed_bracket_released(self):
        f = GestureTagFilter({"wave"})
        long = "[" + "x" * 60
        self.assertTrue((f.feed(long) + f.flush()).startswith("["))

    def test_gesture_fires_once(self):
        got = []
        f = GestureTagFilter({"wave"}, on_gesture=got.append)
        f.feed("[gesture:wave] and again [gesture:wave]")
        self.assertEqual(got, ["wave"])


class TestRouterAndGrammar(unittest.TestCase):
    kw = ["see", "holding", "how many", "color"]

    def test_vision_router(self):
        self.assertTrue(needs_vision("What am I holding?", self.kw))
        self.assertTrue(needs_vision("How many people are here", self.kw))
        self.assertFalse(needs_vision("Tell me a joke", self.kw))
        self.assertFalse(needs_vision("seesaw", self.kw))

    def grammar(self, wake=True):
        rules = [{"pattern": r"\bwave\b", "action": "gesture", "arg": "wave"},
                 {"pattern": r"\bstep (to the )?left\b", "action": "step", "arg": "left"}]
        return CommandGrammar(rules, ["stop", "freeze"], ["robot"], wake)

    def test_stop_needs_no_wake_word_and_excludes_motion(self):
        g = self.grammar()
        self.assertEqual(g.match("stop waving"), [("stop", "")])

    def test_motion_requires_wake_word(self):
        g = self.grammar()
        self.assertEqual(g.match("wave"), [])
        self.assertEqual(g.match("robot wave"), [("gesture", "wave")])
        self.assertEqual(g.match("robot wave"), [])          # once per utterance
        g.reset()
        self.assertEqual(g.match("Robot, step to the left"), [("step", "left")])

    def test_wake_word_optional(self):
        self.assertEqual(self.grammar(False).match("wave please"), [("gesture", "wave")])

    def test_command_only(self):
        g = self.grammar()
        self.assertTrue(g.is_command_only("Robot, wave please."))
        self.assertFalse(g.is_command_only("Robot, wave and then tell me about the weather in Paris"))
        self.assertTrue(has_wake_word("hey Robot!", ["robot"]))
        self.assertFalse(has_wake_word("robotics", ["robot"]))


if __name__ == "__main__":
    unittest.main()
