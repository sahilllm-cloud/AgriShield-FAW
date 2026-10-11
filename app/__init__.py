"""AgriShield backend package."""

# Load PyTorch before Uvicorn imports app.main. On Windows this avoids a
# c10.dll initialization failure caused by the Uvicorn CLI import path.
import torch
