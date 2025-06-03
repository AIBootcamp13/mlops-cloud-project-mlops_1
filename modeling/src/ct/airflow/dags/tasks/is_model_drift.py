def is_model_drift():
    import glob
    import os
    import pandas as pd

    DRIFT_THRESHOLD = 50

    anomalies_files = glob.glob("dags/data/outputs/inference/*_anomalies.csv")
    anomalies_files.sort(key=os.path.getmtime)
    latest_anomalies_file = anomalies_files[-1]

    df = pd.read_csv(latest_anomalies_file)

    return len(df) > DRIFT_THRESHOLD