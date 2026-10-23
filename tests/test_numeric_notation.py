"""Fixed-input verifier regressions; no model or provider invocation."""

import unittest

from agent_loop.memory import Memory
from agent_loop.models import ToolCall
from agent_loop.verify import extract_numbers, verify_grounding


def memory_with_results(payload):
    memory = Memory()
    memory.add_tool_results([(ToolCall(id="fixed", name="calculate", input={}), payload)])
    return memory


class NumericNotationTests(unittest.TestCase):
    def test_scientific_and_leading_decimal_notation(self):
        self.assertEqual(extract_numbers("1e3; 2.5e-3; .5; -4E+2"), [1000.0, 0.0025, 0.5, -400.0])

    def test_accepts_equivalent_scientific_tool_result(self):
        result = verify_grounding("fixed", "The result is 1e3", memory_with_results({"result": 1000}))
        self.assertTrue(result.ok, result.feedback)

    def test_rejects_scientific_claim_not_present_in_results(self):
        result = verify_grounding("fixed", "The result is 1e3", memory_with_results({"values": [1, 3]}))
        self.assertFalse(result.ok, "The numeric claim 1000 must not be grounded by unrelated 1 and 3")

    def test_existing_hyphenated_identifier_is_not_numeric(self):
        self.assertEqual(extract_numbers("pump-7 and S7-1500"), [])


if __name__ == "__main__":
    unittest.main()
