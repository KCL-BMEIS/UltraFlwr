# Training Workflow

## SSL Pretraining

```bash
fedyolo-train-ssl
```

**Output:**
- Client-specific SSL weights: `/experiments/ssl_pretraining/checkpoints/ssl_clients/client_X/ssl_backbone.pt`
- Server-aggregated SSL weights: `/weights/ssl/yolo11n-ssl.pt`

## Supervised Training

```bash
fedyolo-train
```

**Initialization (Before Round 1):**
- **Server**: Loads server-aggregated SSL weights once at startup
- **Clients**: Each client loads SSL weights ONCE on first call:
  - First preference: Client-specific SSL weights from `/experiments/ssl_pretraining/checkpoints/ssl_clients/client_X/ssl_backbone.pt`
  - Fallback: Server-aggregated SSL weights from `/weights/ssl/yolo11n-ssl.pt`
  - A marker file (`.ssl_loaded`) is created to prevent redundant loading

**Training Flow:**
- **Round 1 (first fit call)**:
  - Clients load SSL weights from disk (one-time initialization)
  - Marker file created: `.ssl_loaded`
  - Both clients and server start training with SSL-pretrained backbones
  - **Within the round**: Clients train locally for configured epochs, updating weights through gradient descent
  - After training: Clients send updated parameters to server for aggregation
- **Round 1 (evaluate) & Round 2+**:
  - Clients skip SSL loading (marker file exists)
  - Initialize model architecture only (no weight loading from disk)
  - Receive aggregated parameters from server via `set_parameters()`
  - **Within each round**: Clients train locally for configured epochs, updating weights through gradient descent
  - After training: Clients send updated parameters to server for aggregation
  - Standard federated averaging continues

**Key Behavior:**
- SSL weights are loaded from disk **exactly ONCE** per client per training run (first fit call only)
- After first load, clients **do not load any weights from disk**
- All subsequent model updates come **only from server aggregation** (between rounds)
- **Within each round**, client weights are updated normally through local training epochs
- Marker files ensure no redundant SSL weight loading across Flower's process spawns

## Testing/Evaluation

```bash
fedyolo-test
```

**What it does:**
- Evaluates trained models on test data
- Tests each client's final model (from Round 2) on their own test data
- Generates performance metrics and visualizations

**Output:**
- **Metrics**: `/experiments/FedBackboneAvg/results/metrics/client_X_own_data_*.csv`
  - Detection tasks: precision, recall, mAP50, mAP50-95 per class
  - Segmentation tasks: box and mask metrics
  - Pose tasks: box and pose keypoint metrics
  - Classification tasks: top1 and top5 accuracy
- **Visualizations**: `/experiments/FedBackboneAvg/results/visualizations/client_X/`
  - Confusion matrices (normalized and regular)
  - Precision/Recall/F1 curves
  - Prediction visualizations (ground truth vs predictions)

**Testing Modes:**
- **Client-Client**: Each client model tested on its own data (default for multi-task FL)
- **Cross-client**: Test model on other clients' data (requires same task type)

## Complete Workflow

```
SSL Pretraining → Supervised FL Training → Testing/Evaluation
```

Each client benefits from:
1. Their own local SSL pretraining (personalized initialization)
2. Global federated learning aggregation (knowledge sharing)
3. Comprehensive evaluation metrics and visualizations
