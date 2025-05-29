import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import fire
from dotenv import load_dotenv

from modeling.src.main import run_train, run_inference

def main(run_mode, data_root_path, model_root_path):
    load_dotenv()
    if run_mode == "train":
        run_train(data_root_path, model_root_path)
    elif run_mode == "inference":
        run_inference(data_root_path, model_root_path)

if __name__ == '__main__':

    fire.Fire(main)
    
