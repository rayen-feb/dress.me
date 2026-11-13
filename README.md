# DressMe

DressMe is a local web app for organizing a wardrobe, planning outfits for events, and keeping a shopping list. It uses a FastAPI server and a plain HTML/CSS/JavaScript interface.

## Features

- **Wardrobe scan:** Upload one garment photo to get suggested garment type, colour, and pattern. Review the suggestions before saving.
- **Duplicate detection:** Compare a new scan with saved wardrobe photos to find visually similar items.
- **Outfit recommendations:** Describe an event to get outfit ideas based on saved garments, dress code, setting, and weather.
- **Calendar:** Add, edit, and delete events in a month view. Use a saved event to fill in the outfit recommendation form.
- **Shopping:** Get wardrobe-gap suggestions and add items to a personal list. Mark items as bought or remove them.

## Requirements

- Python 3.10 or newer
- Internet access the first time the app starts, to download the Fashion-CLIP model
- Internet access for weather forecasts; the app uses [Open-Meteo](https://open-meteo.com/)

## Run on Windows

Open PowerShell in the project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn server:app --reload --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Keep the PowerShell window open while using the app; press **Ctrl+C** to stop the server.

If creating the virtual environment fails with `No module named venv`, install or repair a standard Python installation that includes `venv`, then repeat the setup commands.

### Run on macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn server:app --reload --port 8000
```

The Fashion-CLIP model (`patrickjohncyh/fashion-clip`) is loaded by Transformers and may take a while to download and initialize on first launch. Scanning can be slower on a CPU.

## Using the app

1. In **Wardrobe**, upload a JPEG, PNG, or WebP photo of one garment (up to 15 MB).
2. Review and correct the suggested details, then save the garment.
3. Use **Calendar** to save an event, or open **Outfits** to request recommendations directly.
4. In **Shopping**, add a suggested wardrobe gap to your list or enter an item yourself.

Predictions are suggestions and can be corrected. Photos with one clearly visible garment and a simple background work best.

## Local data and privacy

The app stores wardrobe details and photos, calendar events, and shopping-list entries under `data/` on the machine running the server. This directory is intentionally excluded from Git so personal wardrobe data is not published with the source code.

This demo has no accounts or authentication and is intended for local use. Do not expose the development server to the public internet. When requesting outfit ideas, the event city is sent to Open-Meteo for a forecast. Pinterest is opened only when you choose its inspiration link.

## Project layout

```text
server.py                         FastAPI app and API routes
index.html                        Web interface
requirements.txt                  Python dependencies
dso1_wardrobe_scan/
  models/                         Classifier and attribute data used by the scanner
  src/                            Image preprocessing and inference
dso1_01_data_exploration.py        Colab-exported model/data exploration
data/                             Local wardrobe, photos, calendar, and list (not tracked)
```

`dso1_01_data_exploration.py` is a research artifact, not part of the web app startup. It was exported from Colab and contains notebook commands; open it in a compatible notebook environment rather than running it directly with `python`. It downloads external Kaggle datasets and needs its own packages and Kaggle access. Follow the dataset's terms and never commit Kaggle credentials.
