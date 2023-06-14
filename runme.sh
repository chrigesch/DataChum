#!/usr/bin/bash
cd /home/ceci/Proyectos/DataChum
source /home/ceci/miniconda3/bin/activate datachum_39
nohup streamlit run index.py >/dev/null 2>&1 &
xdg-open http://192.168.1.4:8501
