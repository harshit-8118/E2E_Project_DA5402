# run this on linux machine to save weights for cpu machine. 
import mlflow
import torch
print("Loading production model...")

model_save_dir = "outputs/models/best_model_cpu.pth"
prod = mlflow.pytorch.load_model(
    "models:/skin-disease-classifier@production"
)

# if wrapped model
base_model = prod.model if hasattr(prod, "model") else prod

base_model.eval()

torch.save(
    base_model.state_dict(),
    model_save_dir
)

print("Saved clean weights:")
print(f"{model_save_dir}")