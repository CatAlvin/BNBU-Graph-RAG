import unittest
from agents.rag_specialist_agent import Claim, EvidenceSpan, RAGSpecialistOutput, ground_citations

class EvidenceTests(unittest.TestCase):
    def pack(self, quote, source="VECTOR", page=None):
        return RAGSpecialistOutput(answer_draft="Closes at 22:00.", claims=[Claim(claim="Closes at 22:00.", evidence=[EvidenceSpan(source=source, page=page, text=quote)])])

    def test_recovers_filename_and_page_from_exact_quote(self):
        result = ground_citations(self.pack("closes at 22:00"), [{"source":"sample-campus.pdf", "page":1, "text":"The library closes at 22:00 on weekdays."}])
        self.assertEqual((result.claims[0].evidence[0].source, result.claims[0].evidence[0].page), ("sample-campus.pdf", 1))

    def test_fabricated_quote_cannot_keep_a_valid_looking_filename(self):
        result = ground_citations(self.pack("closes at midnight", "sample-campus.pdf", 1), [{"source":"sample-campus.pdf", "page":1, "text":"Closes at 22:00."}])
        self.assertTrue(result.recommend_refuse)
        self.assertEqual(result.claims, [])

    def test_ambiguous_quote_does_not_get_an_arbitrary_citation(self):
        result = ground_citations(self.pack("closes at 22:00"), [{"source":name, "page":1, "text":"Closes at 22:00."} for name in ["a.pdf", "b.pdf"]])
        self.assertTrue(result.recommend_refuse)
        self.assertEqual(result.claims, [])

if __name__ == "__main__":
    unittest.main()
