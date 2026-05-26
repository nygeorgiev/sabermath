import random
from datasets import load_dataset, Dataset


TARGETS_DATASET = "sabermath/SaberMath-queries"
CANDIDATES_DATASET = "sabermath/SaberMath-documents"


def load_data() -> tuple[Dataset, Dataset]:
    targets = load_dataset(TARGETS_DATASET, split="train")
    candidates = load_dataset(CANDIDATES_DATASET, split="train")
    return targets, candidates
