# AdaptiveViewSelection

folgende Einstellungen für
.vscode\settings.json
{
    "python-envs.defaultEnvManager": "ms-python.python:system"
}

folgende Einstellungen für
.vscode\launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: MeshroomImageSelector",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/MeshroomImageSelector.py",
            "console": "integratedTerminal",
            "args": [
                "C:/Users/user/Documents/Master/FlightScanner2/RohData/Holzstamm_jpeg",
                "./selected_60p_70c",
                "--ratio",
                "0.6",
                "--sharpness",
                "0.7"
            ]
        }
    ]
}

wobei args[0] source path
args[1] target path
