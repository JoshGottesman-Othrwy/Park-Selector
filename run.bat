@echo off
REM Launch the Park Meetup Selector Streamlit app.
cd /d "%~dp0"
python -m streamlit run app.py
pause
