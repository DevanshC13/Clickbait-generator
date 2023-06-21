@echo off

REM Create the virtual environment
py -m venv venv

REM Install the required packages
.\venv\Scripts\pip install -r requirements.txt

REM Run the script
.\venv\Scripts\python api.py

