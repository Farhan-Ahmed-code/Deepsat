import subprocess
import sys
import os

if __name__ == "__main__":
    print("[+] Running EuroSAT Baseline CNN model evaluation on the full test dataset...")
    # Get current python executable and run evaluate_baseline.py
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate_baseline.py")
    result = subprocess.run([sys.executable, script_path])
    sys.exit(result.returncode)
