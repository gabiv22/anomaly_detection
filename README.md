# anomaly_detection
to run app : \
pip install streamlit\
streamlit run app.py

upload csv file similar to the original csv file 
or run analysis on the original csv file ( the default)

configurable parameters:
- confidence interval
- split ratio: save last p% of data for anomaly prediction
see requirments.txt for my versions

model will process data row by row.
if model identifies an anomaly - will display the entire row.

terminal willshow additional info
