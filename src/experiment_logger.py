import os
import json
from typing import Dict, Optional

LOG_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/metrics.json')


def load_logs():
    try:
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_logs(logs):
    exp_dir = os.path.dirname(LOG_FILE)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)


def log_experiment(model_name: str,
                   description: Optional[str],
                   eval_metrics: Dict[str, any]):
    logs = load_logs()

    logs[model_name] = {
        "description": description,
        **eval_metrics
    }

    save_logs(logs)
