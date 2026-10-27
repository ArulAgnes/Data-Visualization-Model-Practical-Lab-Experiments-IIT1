# Runs the dataset generator + all 10 simple experiments in order
import subprocess, sys
files = ['00_generate_dataset_simple.py'] + [f'exp{i}_simple.py' for i in range(1, 11)]
for f in files:
    print(f'\n=== Running {f} ===')
    subprocess.run([sys.executable, f], check=True)
print('\nAll done! CSV and PNG files created in this folder.')
