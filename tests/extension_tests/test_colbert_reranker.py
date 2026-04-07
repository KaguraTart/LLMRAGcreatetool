import unittest

from src.extensions.builtin.colbert_reranker import ColBERTRerankerExtension


class ColBERTRerankerTests(unittest.TestCase):
    def test_rerank_promotes_overlap(self):
        reranker = ColBERTRerankerExtension()
        candidates = [
            {"id": "1", "content": "apple banana", "score": 0.2},
            {"id": "2", "content": "orange pear", "score": 0.9},
        ]
        output = reranker.rerank(query="apple", candidates=candidates, top_n=2)
        self.assertEqual(output[0]["id"], "1")
        self.assertIn("colbert_score", output[0])


if __name__ == "__main__":
    unittest.main()
