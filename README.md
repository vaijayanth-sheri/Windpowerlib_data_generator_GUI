# Windpowerlib Dashboard


A clean, open-source dashboard built with Streamlit to simulate wind turbine energy production using weather data and **windpowerlib** (https://github.com/wind-python/windpowerlib). Supports multiple weather sources, turbine models, and full PDF report export.


## ✨ Features
- Fetch & normalize weather data from:
- Open-Meteo (Archive)
- NASA POWER (Hourly, RE community)
- PVGIS TMY (via `pvlib`)
- Custom upload: EPW format (EnergyPlus files)
- Use built-in turbine library (OEDB) or upload custom power curve
- Simulate wind turbine output with **ModelChain**
- Export CSVs of weather and output
- Download a professional PDF report


## ⚙ Installation
```bash
git clone https://github.com/vaijayanth-sheri/Windpowerlib_data_generator_GUI.git 
cd Windpowerlib_data_generator_GUI
pip install -r requirements.txt
```


## 📅 Run it
```bash
streamlit run app.py
```


## 📖 Files
 Windpowerlib_data_generator_GUI <br/>
├─ app.py                      ← main Streamlit app <br/> 
├─ report_wind.py             ← PDF report generator <br/>
├─ normalize.py               ← normalize data to windpowerlib format <br/>
├─ physics.py                 ← optional helper (e.g., uv → speed/dir) <br/>
├─ user_upload.py             ← EPW upload + parsing <br/>
├─ datasources_open_meteo.py ← Open-Meteo adapter <br/>
├─ datasources_power.py      ← NASA POWER adapter <br/>
├─ datasources_pvgis.py      ← PVGIS adapter (via pvlib) <br/>
├─ requirements.txt          ← pinned dependencies <br/>
├─ README.md                 ← full deployment + usage guide <br/>
<br/> 
## 👨‍💼 Developer
Built by **Vaijayanth Sheri** as a reproducible wind modeling tool.


## 🌍 License
MIT License.

---
