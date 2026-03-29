

```
data collection -> data exploration -> data cleaning 
-> model selection & training -> API -> 
frontend -> monitoring -> contaninerized

can use vercel or any free web service for frontend/backend. 
```

PROJECT TITLE
Skin Disease Detection and Clinical Advisory System

1. PROBLEM STATEMENT
This project builds a complete, production-grade AI web application that accepts a skin image from the user, classifies the disease using a deep learning model, and provides disease-specific advisory information including symptoms, possible remedies, and links to trustworthy medical sources. The system is designed and implemented following the full MLOps lifecycle as per DA5402 course guidelines.

2. DATASET
Primary Dataset  : HAM10000
                   - 10,015 dermoscopic images across 7 disease classes
                   - Publicly available on Harvard Dataverse and Kaggle
                   - Classes: Melanoma, Melanocytic Nevi, Basal Cell Carcinoma,
                     Actinic Keratosis, Benign Keratosis, Dermatofibroma,
                     Vascular Lesions

Supplementary    : Additional Kaggle dermoscopy datasets
                   - Multiple datasets from the same domain will be merged
                   - Target label space of 10+ disease classes
                   - Dataset selection will be finalized based on class overlap
                     and annotation quality

Data Versioning  : DVC with DagsHub remote (fallback: Google Drive, chosen based on final dataset size and consistency requirements)

3. MACHINE LEARNING
Model    : CNN (EfficientNetB3 / ResNet50) or Vision Transformer (ViT), fine-tuned on the combined dataset. Architecture selection will be determined by MLflow experiment comparison.

Training : GPU based training given the large dataset size.
                   Class imbalance will be handled via weighted loss functions
                   and targeted data augmentation (flip, rotation, CutMix).

Advisory Layer : Post-classification generative response module that returns a structured advisory card per predicted disease — symptoms, recommended specialist type, remedy suggestions, and links to verified medical sources (e.g., AAD, NHS, PubMed).

Explainability   : Grad-CAM heatmap overlaid on the input image to show which skin region influenced the prediction.

Experiment Tracking : MLflow will track all runs — hyperparameters (learning rate, batch size, frozen layers), metrics (macro F1, per-class AUC, sensitivity, specificity), and artifacts (confusion matrix, Grad-CAM samples, model weights). Manual logging will supplement autolog for domain-specific metrics.

Data Pipeline : Complete data pipeline managed via DVC pipeline stages: prepare → augment → train → evaluate → explain → export Every reproducible run maps to a Git commit hash and a corresponding MLflow Run ID.

CI/CD : Managed through DVC + Git + MLflow. Model is promoted to the registry only if acceptance criterion is met (macro F1 >= 0.75 on held-out test set)[threshold is not fixed at pre stage].

Monitoring : Prometheus instrumentation on the inference service tracking request count, inference latency (p50/p95/p99), confidence score distribution, high-risk prediction alerts, and system resource usage (CPU, memory). Grafana dashboards will visualize all metrics in near-real-time. Alertmanager will notify if error rate exceeds variable(5%) or data drift is detected.

4. SOFTWARE ENGINEERING
Frontend : Web UI where the user uploads a skin image and receives:
                   - Predicted disease class with confidence scores
                   - Grad-CAM heatmap highlighting the affected region
                   - Advisory card with symptoms, remedies, and source links
                   - Thumb up / thumb down feedback on the prediction

Backend  : FastAPI service exposing the following endpoints:
                   POST /predict   — image in, disease class + advisory out
                   POST /explain   — returns Grad-CAM heatmap overlay
                   POST /feedback  — stores user feedback with prediction UUID
                   GET  /health    — Docker health check
                   GET  /ready     — readiness probe
                   GET  /classes   — disease class metadata

Architecture     : Strictly loosely coupled. Frontend communicates with backend only via configurable REST API calls. No shared code or direct database access from the UI layer.

Database : Non-relational database (MongoDB) to store:
                   - User images tagged with UUID (for future retraining)
                   - Minimal login details to prevent false business claims
                   - User feedback responses pending verification

Feedback Loop    : Thumb up/down responses and optional user-reported disease confirmations are collected and quarantined. After human review and verification, confirmed samples are added to the next model retraining cycle.

Containerization : Docker Compose with independent services:
                   frontend | backend | mlflow | prometheus | grafana

Data Engineering : Apache Airflow DAG (or Spark) to automate ingestion,
                   schema validation, missing-value checks, and drift baseline
                   computation (mean, variance, class distribution per feature).


5. DOCUMENTATION DELIVERABLES

  - Architecture diagram with HLD block description
  - High-level design document (design choices and rationale)
  - Low-level design document (all API endpoint I/O specifications)
  - Test plan, test cases, and test report (pytest — unit + integration)
  - User manual for non-technical users


6. ETHICAL CONSIDERATION

All predictions will be accompanied by a disclaimer: "This tool provides AI-assisted preliminary assessment only. It is not a substitute for professional medical diagnosis. Please consult a qualified dermatologist."

User data will be stored securely, used only for verified model improvement, and never shared with third parties.
