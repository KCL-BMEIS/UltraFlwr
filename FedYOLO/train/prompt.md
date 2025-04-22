# Context

Currently, this repo does only training (lets forget about testing for now) for detection. 

Essentially through `net.train(data=data_path, epochs=YOLO_CONFIG['epochs'], workers=0, seed=cid, batch=YOLO_CONFIG['batch_size'], project=strategy)` in `FedYOLO/train/yolo_client.py` .

What I want to do now is also include segmentation.

The code for instance segmentation woudl be something like:

```python
!yolo task=detect mode=train model=yolo11n-seg.pt data={dataset.location}/data.yaml epochs=10 imgsz=640 plots=True
```

Notice, the only change is seg.pt. So we need to load segmentation weights and not the normal weights.

# Exact Ask

Where do I have to make this change?

Looking at the code, I think we need to change nothing else apart from this because everything else stays the same. 