import json
import random
from typing import Dict
import os

def generate_train_predict_dataset(
    num_satellites: int,
    num_ground: int,
    num_train_queries: int,
    num_train_slots: int,
    seed_train: int = 0,
    num_predict_queries: int = 10,
    num_predict_slots: int = 10,
    seed_predict: int = 0,
) -> Dict:
    """Generate a dataset for prediction.

    ``sat_positions_per_slot`` stores the coordinates of all satellites for each
    time slot. Ground station positions remain unchanged.
    """

    random.seed(seed_train)

    # Generate satellite positions for every slot
    sat_positions_per_slot = []
    for _ in range(num_train_slots):
        slot_pos = [
            [random.uniform(0, 100), random.uniform(0, 100)]
            for _ in range(num_satellites)
        ]
        sat_positions_per_slot.append(slot_pos)

    dataset = {
        "num_satellites": num_satellites,
        "num_ground": num_ground,
        "num_train_slots": num_train_slots,
        "num_predict_slots": num_predict_slots,
        "num_train_queries": num_train_queries,
        "num_predict_queries": num_predict_queries,

        "sat_positions_per_slot": sat_positions_per_slot,
        "gs_positions": [
            [random.uniform(0, 100), random.uniform(0, 100)]
            for _ in range(num_ground)
        ],
        "train_queries": [],
        "predict_queries": [],
    }

    for _ in range(num_train_queries):
        src = random.randint(0, num_ground - 1)
        dst = random.randint(0, num_ground - 1)
        while dst == src:
            dst = random.randint(0, num_ground - 1)
        time_slot = random.randint(0, num_train_slots - 1)
        dataset["train_queries"].append({"src": src, "dst": dst, "time": time_slot})

    dataset["train_queries"].sort(key=lambda q: q["time"])

    for _ in range(num_predict_queries):
        src = random.randint(0, num_ground - 1)
        dst = random.randint(0, num_ground - 1)
        while dst == src:
            dst = random.randint(0, num_ground - 1)
        time_slot = random.randint(0, num_predict_slots - 1)
        dataset["predict_queries"].append({"src": src, "dst": dst, "time": time_slot})

    return dataset


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    num_satellites = config.get("num_satellites", 5)
    num_ground = config.get("num_ground_stations", 5)
    
    # train queries and slots:
    # train query: train["queries"]
    train_config = config.get("train", {})
    num_train_queries = train_config.get("train_queries", 10)
    num_train_slots= train_config.get("max_time", 10)

    predict_config = config.get("predict", {})
    num_predict_queries = predict_config.get("predict_queries", 10)
    num_predict_slots = predict_config.get("max_time", 10)

    # Generate dataset with default parameters or from config
    data = generate_train_predict_dataset(
        num_satellites=num_satellites,
        num_ground=num_ground,
        num_train_queries=num_train_queries,
        num_train_slots=num_train_slots,
        seed_train=train_config.get("train_seed", 0),
        num_predict_queries=num_predict_queries,
        num_predict_slots=num_predict_slots,
        seed_predict=predict_config.get("predict_seed", 0),
    )

    #data_path = config.get("data_dir", "data")+config.get("data_name", "prediction_data.json")+ ".json"
    data_path = os.path.join(
        config.get("data_dir", "data"),
        f"{config.get('data_name', 'prediction_data')}.json"
    )
    with open(data_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Generated data saved to {data_path}")
