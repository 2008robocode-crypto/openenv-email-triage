import time
import subprocess

print("Starting evaluation...")

subprocess.run(["python", "run_eval.py"])

print("Evaluation finished. Keeping container alive.")

while True:
    time.sleep(60)