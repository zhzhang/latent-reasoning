import unittest

from alt_test import score_binary_judge


class ScoreBinaryJudgeTests(unittest.TestCase):
    def test_preserves_stable_annotator_ids_across_varying_teams(self) -> None:
        people = ("alice", "bob", "carol", "dave")
        instances = []
        for index in range(40):
            present = [
                person
                for person_index, person in enumerate(people)
                if person_index != index % len(people)
            ]
            ratings = {
                person: (index + rating_index) % 2 == 0
                for rating_index, person in enumerate(reversed(present))
            }
            instances.append((f"item-{index}", ratings, index % 2 == 0))

        result = score_binary_judge(instances)

        self.assertEqual(result["n_annotators"], 4)
        self.assertEqual(set(result["per_annotator"]), set(people))
        self.assertEqual(
            {details["n"] for details in result["per_annotator"].values()},
            {30},
        )

    def test_supports_legacy_positional_ratings(self) -> None:
        instances = [
            (f"item-{index}", [True, False, True], True)
            for index in range(30)
        ]

        result = score_binary_judge(instances)

        self.assertEqual(result["n_annotators"], 3)
        self.assertEqual(set(result["per_annotator"]), {"0", "1", "2"})


if __name__ == "__main__":
    unittest.main()
