import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help="Path to data")
parser.add_argument("--model", type=str, required=True, help="Path to model")
args = parser.parse_args()

model = YOLO(args.model)

metrics = model.val(data=args.data, split="test", project="local_test")