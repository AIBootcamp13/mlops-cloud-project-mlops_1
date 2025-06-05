import os
import json

vscode_dir = ".vscode"
settings_path = os.path.join(vscode_dir, "settings.json")

settings_data = {
    "python.analysis.extraPaths": [
        "./airflow/utils"
    ],
    "python.linting.enabled": True,
    "python.analysis.diagnosticMode": "workspace",
    "python.analysis.useLibraryCodeForTypes": True
}

# .vscode 폴더 없으면 생성
if not os.path.exists(vscode_dir):
    os.makedirs(vscode_dir)

# settings.json 저장
with open(settings_path, "w") as f:
    json.dump(settings_data, f, indent=2)

print(f"VS Code 설정 파일이 생성되었습니다: {settings_path}")