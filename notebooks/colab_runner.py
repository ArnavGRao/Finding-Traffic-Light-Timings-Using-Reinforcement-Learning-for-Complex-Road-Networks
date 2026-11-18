# %% [markdown]
# # D3QN Traffic Signal Control — Google Colab Headless Training & 50-Seed Evaluation

# %% Install SUMO and Python Dependencies
import os, subprocess

subprocess.run("add-apt-repository -y ppa:sumo/stable && apt-get update -qq && apt-get install -y -qq sumo sumo-tools", shell=True, check=True)
subprocess.run("pip install -q sumo-rl gymnasium numpy pandas", shell=True, check=True)
os.environ["SUMO_HOME"] = "/usr/share/sumo"

# %% Mount Google Drive & Sync Repository
from google.colab import drive
drive.mount("/content/drive")

REPO_URL = "https://github.com/<YOUR_ORG_OR_USER>/Finding-Traffic-Light-Timings-Using-Reinforcement-Learning-for-Complex-Road-Networks.git"
REPO_DIR = "/content/d3qn-traffic"
DRIVE_OUT = "/content/drive/MyDrive/d3qn_traffic_outputs"
os.makedirs(DRIVE_OUT, exist_ok=True)

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "-b", "feat/d3qn-agent", REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)

os.chdir(REPO_DIR)

# %% Train All 5 Ablation Variants (Auto-Resumes from Drive if Disconnected)
MODELS = ["vanilla_dqn", "model_a", "model_b", "model_c", "model_d"]
REGIME = "balanced"
ROUTE_FILE = "networks/single_intersection/single-intersection-vhvh.rou.xml"

for model_name in MODELS:
    cmd = (
        f"PYTHONPATH=. python3 scripts/train_rl.py "
        f"--model {model_name} --regime {REGIME} "
        f"--route-file {ROUTE_FILE} "
        f"--episodes 40 --num-seconds 3600 "
        f"--output-dir {DRIVE_OUT}/runs --resume"
    )
    print("Running:", cmd)
    subprocess.run(cmd, shell=True, check=True)

# %% Run 50-Seed Deterministic Evaluation for All Trained Models
for model_name in MODELS:
    ckpt = f"{DRIVE_OUT}/runs/{model_name}_{REGIME}/best.pt"
    cmd = (
        f"PYTHONPATH=. python3 scripts/evaluate_rl.py "
        f"--model {model_name} --regime {REGIME} "
        f"--checkpoint {ckpt} "
        f"--route-file {ROUTE_FILE} "
        f"--num-seeds 50 --num-seconds 3600 "
        f"--output-dir {DRIVE_OUT}/eval"
    )
    print("Evaluating:", cmd)
    subprocess.run(cmd, shell=True, check=True)
