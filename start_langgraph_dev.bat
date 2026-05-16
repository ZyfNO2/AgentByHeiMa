@echo off
call G:\Anaconda\Scripts\activate.bat Agent
python -u -m langgraph_cli dev --no-browser --port 8123
pause
