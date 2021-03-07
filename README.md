## Group: wewhoshallnotbenamed

Repository for the University of Helsinki course KIK-LG211 Building NLP Applications course work.
 
# Final project: Wine search 
## Description

## How to use

1) Clone the project by `git clone `

2) You can create virtual environment by `python3 -m venv projectenv` (windows: `py -3 -m venv projectenv`). If you do not
have Python installed, see [Python installation](https://www.python.org/downloads/).

3) Activate environment `. projectenv/bin/activate` (windows: `projectenv/Scripts/activate`)

4) `cd` to **wewhoshallnotbenamed** directory

5) Install the project dependencies by `pip install -r requirements.txt` (or `pip3 install -r requirements.txt`)

6) Install PKE following these [instructions](https://github.com/boudinfl/pke)

7) To run the final project, `cd` to **final_project** directory

8) Provide the following environment variables:

```
export FLASK_APP=ui.py  
export FLASK_RUN_PORT=8000  
```

Windows:

```
set FLASK_APP=ui.py  
set FLASK_ENV=development  
set FLASK_RUN_PORT=8000  
```

Windows Powershell:

```
$env:FLASK_APP = "ui.py"  
$env:FLASK_ENV = "development"  
$env:FLASK_RUN_PORT = "8000"  
```

9) Execute `flask run` and go to `localhost:8000/search` to see the project on browser.

## Known issues
 
## Resources used in final project:
- Wine review data for final project is from https://www.kaggle.com/zynicide/wine-reviews?select=winemag-data-130k-v2.csv
- Flag icons are from https://www.countryflags.io/
- Wine icon is from https://www.pngfind.com/mpng/hwbmiTi_icon-png-download-wine-glass-svg-file-free/

## Contributors
Group consists of Essi, Helmiina, Jussi-Veikka and Laura.
