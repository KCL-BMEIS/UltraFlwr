import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help="Path to data")
parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible training")
args = parser.parse_args()

model = YOLO()

# Set the seed for reproducible training
results = model.train(data=args.data, batch=8, epochs=400, project="global_local_train", seed=args.seed)

metrics = model.val(data=args.data, split="test")
