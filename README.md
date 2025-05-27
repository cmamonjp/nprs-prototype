# NRRS-P Prototype v0.8

NRRS-P Prototype v0.8 is a Streamlit app that analyzes FIT files from cycling or trail running activities. It calculates and visualizes power output per kilogram (W/kg) categorized by terrain gradients (uphill, flat, downhill).

## Live Demo  
Try the app online here:  
https://nprs-prototype-bvgby4r8fwcnnjx85mxoio.streamlit.app/

## Features  
- Upload a FIT file containing activity data  
- Calculates gradient based on altitude and distance  
- Classifies segments into uphill, flat, and downhill based on gradient thresholds  
- Calculates power-to-weight ratio (W/kg) based on user input weight  
- Visualizes raw and smoothed W/kg over elapsed time for each terrain type  
- Shows average W/kg by terrain segment  
- Allows CSV export of parsed and processed data

## Requirements  
- Python 3.8 or higher  
- Packages: streamlit, pandas, numpy, matplotlib, fitdecode  
- See `requirements.txt` for exact versions

## Installation and Usage

```bash
git clone https://github.com/cmamonjp/nprs-prototype.git
cd nprs-prototype
pip install -r requirements.txt
streamlit run app.py
