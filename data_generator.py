import json
import random
from typing import List, Dict


def generate_dataset(
    num_satellites: int,
    num_ground: int,
    num_queries: int,
    num_slots: int,
    seed: int = 0,
) -> Dict:
    """Generate a dataset for prediction.

    ``sat_positions_per_slot`` stores the coordinates of all satellites for each
    time slot. Ground station positions remain unchanged.
    """

    random.seed(seed)

    # Generate satellite positions for every slot
    sat_positions_per_slot = []
    for _ in range(num_slots):
        slot_pos = [
            [random.uniform(0, 100), random.uniform(0, 100)]
            for _ in range(num_satellites)
        ]
        sat_positions_per_slot.append(slot_pos)

    dataset = {
        "sat_positions_per_slot": sat_positions_per_slot,
        "gs_positions": [
            [random.uniform(0, 100), random.uniform(0, 100)]
            for _ in range(num_ground)
        ],
        "queries": [],
    }

    for _ in range(num_queries):
        src = random.randint(0, num_ground - 1)
        dst = random.randint(0, num_ground - 1)
        while dst == src:
            dst = random.randint(0, num_ground - 1)
        time_slot = random.randint(0, num_slots - 1)
        dataset["queries"].append({"src": src, "dst": dst, "time": time_slot})

    dataset["queries"].sort(key=lambda q: q["time"])

    return dataset


if __name__ == "__main__":
    data = generate_dataset(5, 5, 10, 50)
    with open("data/prediction_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Generated data saved to data/prediction_data.json")
