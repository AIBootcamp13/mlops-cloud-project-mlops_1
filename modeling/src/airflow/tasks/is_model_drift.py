def is_model_drift(project_path):
    import glob
    import os
    import pandas as pd

    DRIFT_THRESHOLD = 40

    anomalies_files = glob.glob(os.path.join(project_path, "data/outputs/inference/*_temperature_*_anomalies.csv"))
    anomalies_files.sort(key=os.path.getmtime)
    latest_anomalies_file = anomalies_files[-1]

    df = pd.read_csv(latest_anomalies_file)

    return len(df) > DRIFT_THRESHOLD