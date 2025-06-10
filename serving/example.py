import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import fire
from dotenv import load_dotenv

from modeling.src.main import run_train, run_inference

def main(run_mode, data_root_path, model_root_path, batch_size=64):
    load_dotenv()

    if run_mode == "train":
        val_loss = run_train(data_root_path, model_root_path, batch_size)
        print(val_loss)
    elif run_mode == "inference":
        temperature_results, PM_results = run_inference(data_root_path, model_root_path, batch_size)
        print(temperature_results, PM_results)

if __name__ == '__main__':

    fire.Fire(main)
    
