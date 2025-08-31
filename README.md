# Windpowerlib Dashboard


A clean, open-source dashboard built with Streamlit to simulate wind turbine energy production using weather data and **windpowerlib**. Supports multiple weather sources, turbine models, and full PDF report export.


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
git clone https://github.com/YOUR_USERNAME/windpowerlib-dashboard.git
cd windpowerlib-dashboard
pip install -r requirements.txt
```


## 📅 Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```


## 📖 Files
# Windpowerlib Dashboard
# ├─ app.py
# ├─ report_wind.py
# ├─ normalize.py
# ├─ physics.py
# ├─ user_upload.py
# ├─ datasources_open_meteo.py
# ├─ datasources_power.py
# ├─ datasources_pvgis.py
# ├─ requirements.txt
# ├─ README.md

## 👨‍💼 Developer
Built by **Vaijayanth Sheri** as a reproducible wind modeling tool.


## 🌍 License
MIT License.

---
Made with ❤️ and wind.
