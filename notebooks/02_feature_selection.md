```python
import pandas as pd

df = pd.read_parquet('../data/raw/pitching_stats.parquet')
```


```python
# Columns of interest, determined by Optuna optimization:
#Plus Stats: HR/9+
#Statcast pitch movement (horizontal & vertical for each pitch type): CH, CU, FA, FC, FS, KC, KN, SC, SI, SL
#Traditional stats: Age, BB%, Barrel%, Clutch, ERA, FIP, GB%, HR/FB, K%, O-Contact%, O-Swing%, SwStr%, Z-Contact%, Z-Swing%, Zone%
#Engineered features: Last year’s value and 1-year delta for each Statcast pitch movement feature above (36 features total)
```


```python
# Feature selection
# Select base feature columns
pitches=['FA', 'FC', 'FS', 'SI', 'SL', 'CU', 'KC', 'CH', 'SC', 'KN']
fc_plus=['HR/9+']
fc_hmovement=[f'{p}-X (sc)' for p in pitches]
fc_vmovement=[f'{p}-Z (sc)' for p in pitches]
fc_trad=['Age', 'BB%', 'Barrel%', 'Clutch', 'ERA', 'FIP', 'GB%', 'HR/FB', 'K%', 'O-Contact%', 'O-Swing%', 'SwStr%', 'Z-Contact%', 'Z-Swing%', 'Zone%']

feature_columns = fc_plus + fc_hmovement + fc_vmovement + fc_trad
```


```python
# Feature engineering
ps = df.copy()

ps = ps.sort_values(['IDfg', 'Season'])

lag_cols = fc_hmovement + fc_vmovement
new_cols = {}

for col in lag_3yr_cols:
    t1 = ps.groupby('IDfg')[col].shift(1)
    new_cols[f'{col}_t1'] = t1
    new_cols[f'{col}_delta_1yr'] = ps[col] - t1
    feature_columns += [f'{col}_delta_1yr', f'{col}_t1']

ps = pd.concat([ps, pd.DataFrame(new_cols, index=ps.index)], axis=1)
ps['ERA_next'] = ps.groupby('IDfg')['ERA'].shift(-1)
```


```python
# Data cleanup
# Only entries with more than 15 innings pitched
ps = ps[ps['IP']>15]
```


```python
import json
# Save the processed and filtered stats with all of their columns so we can use the columns later if needed
ps.to_parquet('../data/raw/processed_pitch_stats.parquet')
with open('../data/raw/feature_columns.json', 'w') as file:
    json.dump(feature_columns, file, indent=4)
```


```python

```
