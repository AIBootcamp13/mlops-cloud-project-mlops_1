import os
import shutil
import sys

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(project_path)
sys.path.append(project_path)

from modeling.src.airflow.tasks.inference import inference

if __name__ == "__main__":
    inference(project_path)