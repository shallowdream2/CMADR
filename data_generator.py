import json
import random
from typing import List, Dict


def generate_dataset(num_satellites: int, num_ground: int, num_queries: int, seed: int = 0) -> Dict:
    random.seed(seed)
    dataset = {
        "sat_positions": [
            [random.uniform(0, 100), random.uniform(0, 100)]
            for _ in range(num_satellites)
        ],
        "gs_positions": [
            [random.uniform(0, 100), random.uniform(0, 100)]
            for _ in range(num_ground)
        ],
        "queries": []
    }
    for _ in range(num_queries):
        src = random.randint(0, num_ground - 1)
        dst = random.randint(0, num_ground - 1)
        while dst == src:
            dst = random.randint(0, num_ground - 1)
        dataset["queries"].append({"src": src, "dst": dst})
    return dataset


if __name__ == "__main__":
    data = generate_dataset(5, 5, 10)
    with open("data/prediction_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Generated data saved to data/prediction_data.json")
