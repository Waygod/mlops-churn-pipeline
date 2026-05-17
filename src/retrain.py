import subprocess
import sys

COMMANDS = [
    [sys.executable, "src/generate_data.py"],
    [sys.executable, "src/preprocess.py"],
    [sys.executable, "src/train.py"],
    [sys.executable, "src/evaluate.py"],
]

def main() -> None:
    for command in COMMANDS:
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, check=True)
    print("Continuous training completed successfully.")

if __name__ == "__main__":
    main()
