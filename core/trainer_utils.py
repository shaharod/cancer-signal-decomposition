import os
import json
import torch
import numpy as np
from pathlib import Path

class ExperimentIO:
    @staticmethod
    def ensure_dir(*args):
        path = os.path.join(*args)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _to_serializable(o):
        """Recursively convert types for JSON safety."""
        if isinstance(o, np.generic): return o.item()
        if torch.is_tensor(o): 
            return o.detach().cpu().item() if o.numel() == 1 else o.detach().cpu().tolist()
        if isinstance(o, dict): return {k: ExperimentIO._to_serializable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [ExperimentIO._to_serializable(x) for x in o]
        return o

    @staticmethod
    def save_results(data, folder, filename="results.json"):
        clean_data = ExperimentIO._to_serializable(data)
        path = os.path.join(folder, filename)
        with open(path, "w") as f:
            json.dump(clean_data, f, indent=2)

    @staticmethod
    def load_results(folder_path, filename):
        """
        Safely loads a JSON result file. 
        Returns a dictionary if found, else an empty dict or None.
        """
        path = Path(folder_path) / filename
        if not path.exists():
            print(f"--> Warning: File not found at {path}")
            return {}
            
        with open(path, "r") as f:
            return json.load(f)
    @staticmethod
    def save_checkpoint(state_dict, folder, filename="model.pt"):
        """
        Saves the best model weights returned by Trainer.fit().
        Note: We save the state_dict, not the whole object.
        """
        path = os.path.join(folder, filename)
        torch.save(state_dict, path)
        print(f"[Saved] Best Model Weights -> {path}")