## Important
- This repository aims to provide guidance on how to use <code>vitaldb</code> dataset.

> Note that <b>all users who use VitalDB, an open biosignal dataset, must agree to the Data Use Agreement below. 
</b> If you do not agree, please close this window.
Click here: [Data Use Agreement](https://vitaldb.net/dataset/?query=overview&documentId=13qqajnNZzkN7NZ9aXnaQ-47NWy7kx-a6gbrcEsi-gak&sectionId=h.vcpgs1yemdb5)

<hr>


## Version info

|Module|Version|
|------|-------|
|Python|3.9.12|
|Keras|2.9.0|
|Tensorflow|2.9.1|


## Sample Code

|Sample code|Objective|
|-----------|---------|
|[vitaldb_open_dataset](./vitaldb_open_dataset.ipynb)|How to use VitalDB open dataset|
|[vitaldb_python_library](./vitaldb_python_library.ipynb)|How to handle Vital files using vitaldb Python library|
|[predict_mortality](./predict_mortality.ipynb)|Prediction of in-hospital mortality in surgical patients|
|[asa_mortality](./asa_mortality.ipynb)|Compare mortality rates depending on ASA physical status class|
|[xgb_mortality](./xgb_mortality.ipynb)|Prediction of mortality rate using xgboost|
|[vitaldb_quality_check](./vitaldb_quality_check.ipynb)|Check the quality of data from vitalDB|
|[hypotension_mbp](./hypotension_mbp.ipynb)|Hypotension prediction using mean blood pressure values|
|[hypotension_art](./hypotension_art.ipynb)|Hypotension prediction using arterial blood pressure waveform|
|[eeg_mac](./eeg_mac.ipynb)|Prediction of anesthetic concentrationion from EEG|
|[ppf_bis](./ppf_bis.ipynb)|Drug Effect Prediction (Propofpl / Remifentanil)|
|[mit_bih_arrhythmia](./mit_bih_arrhythmia.ipynb)|MIT-BIH Arrhyhmia dataset|
|[mbp_mins](./mbp_mins.ipynb)|Calculation of MINS risk depending on BP during surgery|
|[mbp_aki](./mbp_aki.ipynb)|Calculation of AKI risk depending on BP during surgery|
|[vitaldb_tableone](./vitaldb_tableone.ipynb)|Make a group table depending on the death status|



## Update

|Code Update|Detail|Need to check (V:Completed)|
|-----------|------|---------------------------|
|[mbp_mins.ipynb](./mbp_mins.ipynb)|There is no lab_name 'Troponin I'||
|[vitalfile.ipynb](./vitalfile.ipynb)|Edited 'wget' code, No attribute 'exclude_undefined', 'detect_qrs' in <code>pyvital</code> library||
|[vitaldb_quality_check.ipynb](./vitaldb_quality_check.ipynb)|No attribute 'load_trks' in <code>vitaldb</code> library||
|[hypotension_art.ipynb](./hypotension_art.ipynb)|Tensorflow layer issue| (seems version issue)|
|[vitaldb_tableone.ipynb](./vitaldb_tableone.ipynb)|No columns 'filename','casedur', 'opdur', 'anedur' in 'https://api.vitaldb.net/cases' -> Delete them|V|
|[mbp_aki.ipynb](./mbp_aki.ipynb)|Change lab name 'Creatinine' to 'cr'|V|
|[mit_bih_arrhythmia.ipynb](./mit_bih_arrhythmia.ipynb)|Edit 'wget' and 'unzip' code|V|
|[hypotension_mbp.ipynb](./hypotension_mbp.ipynb)|Edit directory path|V|



