```python
from pybaseball import pitching_stats

# get season-level stats for all qualified pitchers
df = pitching_stats(2015, 2025)
df.to_parquet('../data/raw/pitching_stats.parquet')

```


```python
# Explore the data, what colums do we have to work with
import pandas as pd
import numpy as np

cols = [f"{i:3d}  {c}" for i, c in enumerate(df.columns)]
n_cols = 6
remainder = len(cols) % n_cols
if remainder:
    cols += [""] * (n_cols - remainder)

grid = np.array(cols).reshape(-1, n_cols)
display(pd.DataFrame(grid).style.set_properties(**{"text-align": "left"}).hide(axis="index").hide(axis="columns"))
```


<style type="text/css">
#T_a0baa_row0_col0, #T_a0baa_row0_col1, #T_a0baa_row0_col2, #T_a0baa_row0_col3, #T_a0baa_row0_col4, #T_a0baa_row0_col5, #T_a0baa_row1_col0, #T_a0baa_row1_col1, #T_a0baa_row1_col2, #T_a0baa_row1_col3, #T_a0baa_row1_col4, #T_a0baa_row1_col5, #T_a0baa_row2_col0, #T_a0baa_row2_col1, #T_a0baa_row2_col2, #T_a0baa_row2_col3, #T_a0baa_row2_col4, #T_a0baa_row2_col5, #T_a0baa_row3_col0, #T_a0baa_row3_col1, #T_a0baa_row3_col2, #T_a0baa_row3_col3, #T_a0baa_row3_col4, #T_a0baa_row3_col5, #T_a0baa_row4_col0, #T_a0baa_row4_col1, #T_a0baa_row4_col2, #T_a0baa_row4_col3, #T_a0baa_row4_col4, #T_a0baa_row4_col5, #T_a0baa_row5_col0, #T_a0baa_row5_col1, #T_a0baa_row5_col2, #T_a0baa_row5_col3, #T_a0baa_row5_col4, #T_a0baa_row5_col5, #T_a0baa_row6_col0, #T_a0baa_row6_col1, #T_a0baa_row6_col2, #T_a0baa_row6_col3, #T_a0baa_row6_col4, #T_a0baa_row6_col5, #T_a0baa_row7_col0, #T_a0baa_row7_col1, #T_a0baa_row7_col2, #T_a0baa_row7_col3, #T_a0baa_row7_col4, #T_a0baa_row7_col5, #T_a0baa_row8_col0, #T_a0baa_row8_col1, #T_a0baa_row8_col2, #T_a0baa_row8_col3, #T_a0baa_row8_col4, #T_a0baa_row8_col5, #T_a0baa_row9_col0, #T_a0baa_row9_col1, #T_a0baa_row9_col2, #T_a0baa_row9_col3, #T_a0baa_row9_col4, #T_a0baa_row9_col5, #T_a0baa_row10_col0, #T_a0baa_row10_col1, #T_a0baa_row10_col2, #T_a0baa_row10_col3, #T_a0baa_row10_col4, #T_a0baa_row10_col5, #T_a0baa_row11_col0, #T_a0baa_row11_col1, #T_a0baa_row11_col2, #T_a0baa_row11_col3, #T_a0baa_row11_col4, #T_a0baa_row11_col5, #T_a0baa_row12_col0, #T_a0baa_row12_col1, #T_a0baa_row12_col2, #T_a0baa_row12_col3, #T_a0baa_row12_col4, #T_a0baa_row12_col5, #T_a0baa_row13_col0, #T_a0baa_row13_col1, #T_a0baa_row13_col2, #T_a0baa_row13_col3, #T_a0baa_row13_col4, #T_a0baa_row13_col5, #T_a0baa_row14_col0, #T_a0baa_row14_col1, #T_a0baa_row14_col2, #T_a0baa_row14_col3, #T_a0baa_row14_col4, #T_a0baa_row14_col5, #T_a0baa_row15_col0, #T_a0baa_row15_col1, #T_a0baa_row15_col2, #T_a0baa_row15_col3, #T_a0baa_row15_col4, #T_a0baa_row15_col5, #T_a0baa_row16_col0, #T_a0baa_row16_col1, #T_a0baa_row16_col2, #T_a0baa_row16_col3, #T_a0baa_row16_col4, #T_a0baa_row16_col5, #T_a0baa_row17_col0, #T_a0baa_row17_col1, #T_a0baa_row17_col2, #T_a0baa_row17_col3, #T_a0baa_row17_col4, #T_a0baa_row17_col5, #T_a0baa_row18_col0, #T_a0baa_row18_col1, #T_a0baa_row18_col2, #T_a0baa_row18_col3, #T_a0baa_row18_col4, #T_a0baa_row18_col5, #T_a0baa_row19_col0, #T_a0baa_row19_col1, #T_a0baa_row19_col2, #T_a0baa_row19_col3, #T_a0baa_row19_col4, #T_a0baa_row19_col5, #T_a0baa_row20_col0, #T_a0baa_row20_col1, #T_a0baa_row20_col2, #T_a0baa_row20_col3, #T_a0baa_row20_col4, #T_a0baa_row20_col5, #T_a0baa_row21_col0, #T_a0baa_row21_col1, #T_a0baa_row21_col2, #T_a0baa_row21_col3, #T_a0baa_row21_col4, #T_a0baa_row21_col5, #T_a0baa_row22_col0, #T_a0baa_row22_col1, #T_a0baa_row22_col2, #T_a0baa_row22_col3, #T_a0baa_row22_col4, #T_a0baa_row22_col5, #T_a0baa_row23_col0, #T_a0baa_row23_col1, #T_a0baa_row23_col2, #T_a0baa_row23_col3, #T_a0baa_row23_col4, #T_a0baa_row23_col5, #T_a0baa_row24_col0, #T_a0baa_row24_col1, #T_a0baa_row24_col2, #T_a0baa_row24_col3, #T_a0baa_row24_col4, #T_a0baa_row24_col5, #T_a0baa_row25_col0, #T_a0baa_row25_col1, #T_a0baa_row25_col2, #T_a0baa_row25_col3, #T_a0baa_row25_col4, #T_a0baa_row25_col5, #T_a0baa_row26_col0, #T_a0baa_row26_col1, #T_a0baa_row26_col2, #T_a0baa_row26_col3, #T_a0baa_row26_col4, #T_a0baa_row26_col5, #T_a0baa_row27_col0, #T_a0baa_row27_col1, #T_a0baa_row27_col2, #T_a0baa_row27_col3, #T_a0baa_row27_col4, #T_a0baa_row27_col5, #T_a0baa_row28_col0, #T_a0baa_row28_col1, #T_a0baa_row28_col2, #T_a0baa_row28_col3, #T_a0baa_row28_col4, #T_a0baa_row28_col5, #T_a0baa_row29_col0, #T_a0baa_row29_col1, #T_a0baa_row29_col2, #T_a0baa_row29_col3, #T_a0baa_row29_col4, #T_a0baa_row29_col5, #T_a0baa_row30_col0, #T_a0baa_row30_col1, #T_a0baa_row30_col2, #T_a0baa_row30_col3, #T_a0baa_row30_col4, #T_a0baa_row30_col5, #T_a0baa_row31_col0, #T_a0baa_row31_col1, #T_a0baa_row31_col2, #T_a0baa_row31_col3, #T_a0baa_row31_col4, #T_a0baa_row31_col5, #T_a0baa_row32_col0, #T_a0baa_row32_col1, #T_a0baa_row32_col2, #T_a0baa_row32_col3, #T_a0baa_row32_col4, #T_a0baa_row32_col5, #T_a0baa_row33_col0, #T_a0baa_row33_col1, #T_a0baa_row33_col2, #T_a0baa_row33_col3, #T_a0baa_row33_col4, #T_a0baa_row33_col5, #T_a0baa_row34_col0, #T_a0baa_row34_col1, #T_a0baa_row34_col2, #T_a0baa_row34_col3, #T_a0baa_row34_col4, #T_a0baa_row34_col5, #T_a0baa_row35_col0, #T_a0baa_row35_col1, #T_a0baa_row35_col2, #T_a0baa_row35_col3, #T_a0baa_row35_col4, #T_a0baa_row35_col5, #T_a0baa_row36_col0, #T_a0baa_row36_col1, #T_a0baa_row36_col2, #T_a0baa_row36_col3, #T_a0baa_row36_col4, #T_a0baa_row36_col5, #T_a0baa_row37_col0, #T_a0baa_row37_col1, #T_a0baa_row37_col2, #T_a0baa_row37_col3, #T_a0baa_row37_col4, #T_a0baa_row37_col5, #T_a0baa_row38_col0, #T_a0baa_row38_col1, #T_a0baa_row38_col2, #T_a0baa_row38_col3, #T_a0baa_row38_col4, #T_a0baa_row38_col5, #T_a0baa_row39_col0, #T_a0baa_row39_col1, #T_a0baa_row39_col2, #T_a0baa_row39_col3, #T_a0baa_row39_col4, #T_a0baa_row39_col5, #T_a0baa_row40_col0, #T_a0baa_row40_col1, #T_a0baa_row40_col2, #T_a0baa_row40_col3, #T_a0baa_row40_col4, #T_a0baa_row40_col5, #T_a0baa_row41_col0, #T_a0baa_row41_col1, #T_a0baa_row41_col2, #T_a0baa_row41_col3, #T_a0baa_row41_col4, #T_a0baa_row41_col5, #T_a0baa_row42_col0, #T_a0baa_row42_col1, #T_a0baa_row42_col2, #T_a0baa_row42_col3, #T_a0baa_row42_col4, #T_a0baa_row42_col5, #T_a0baa_row43_col0, #T_a0baa_row43_col1, #T_a0baa_row43_col2, #T_a0baa_row43_col3, #T_a0baa_row43_col4, #T_a0baa_row43_col5, #T_a0baa_row44_col0, #T_a0baa_row44_col1, #T_a0baa_row44_col2, #T_a0baa_row44_col3, #T_a0baa_row44_col4, #T_a0baa_row44_col5, #T_a0baa_row45_col0, #T_a0baa_row45_col1, #T_a0baa_row45_col2, #T_a0baa_row45_col3, #T_a0baa_row45_col4, #T_a0baa_row45_col5, #T_a0baa_row46_col0, #T_a0baa_row46_col1, #T_a0baa_row46_col2, #T_a0baa_row46_col3, #T_a0baa_row46_col4, #T_a0baa_row46_col5, #T_a0baa_row47_col0, #T_a0baa_row47_col1, #T_a0baa_row47_col2, #T_a0baa_row47_col3, #T_a0baa_row47_col4, #T_a0baa_row47_col5, #T_a0baa_row48_col0, #T_a0baa_row48_col1, #T_a0baa_row48_col2, #T_a0baa_row48_col3, #T_a0baa_row48_col4, #T_a0baa_row48_col5, #T_a0baa_row49_col0, #T_a0baa_row49_col1, #T_a0baa_row49_col2, #T_a0baa_row49_col3, #T_a0baa_row49_col4, #T_a0baa_row49_col5, #T_a0baa_row50_col0, #T_a0baa_row50_col1, #T_a0baa_row50_col2, #T_a0baa_row50_col3, #T_a0baa_row50_col4, #T_a0baa_row50_col5, #T_a0baa_row51_col0, #T_a0baa_row51_col1, #T_a0baa_row51_col2, #T_a0baa_row51_col3, #T_a0baa_row51_col4, #T_a0baa_row51_col5, #T_a0baa_row52_col0, #T_a0baa_row52_col1, #T_a0baa_row52_col2, #T_a0baa_row52_col3, #T_a0baa_row52_col4, #T_a0baa_row52_col5, #T_a0baa_row53_col0, #T_a0baa_row53_col1, #T_a0baa_row53_col2, #T_a0baa_row53_col3, #T_a0baa_row53_col4, #T_a0baa_row53_col5, #T_a0baa_row54_col0, #T_a0baa_row54_col1, #T_a0baa_row54_col2, #T_a0baa_row54_col3, #T_a0baa_row54_col4, #T_a0baa_row54_col5, #T_a0baa_row55_col0, #T_a0baa_row55_col1, #T_a0baa_row55_col2, #T_a0baa_row55_col3, #T_a0baa_row55_col4, #T_a0baa_row55_col5, #T_a0baa_row56_col0, #T_a0baa_row56_col1, #T_a0baa_row56_col2, #T_a0baa_row56_col3, #T_a0baa_row56_col4, #T_a0baa_row56_col5, #T_a0baa_row57_col0, #T_a0baa_row57_col1, #T_a0baa_row57_col2, #T_a0baa_row57_col3, #T_a0baa_row57_col4, #T_a0baa_row57_col5, #T_a0baa_row58_col0, #T_a0baa_row58_col1, #T_a0baa_row58_col2, #T_a0baa_row58_col3, #T_a0baa_row58_col4, #T_a0baa_row58_col5, #T_a0baa_row59_col0, #T_a0baa_row59_col1, #T_a0baa_row59_col2, #T_a0baa_row59_col3, #T_a0baa_row59_col4, #T_a0baa_row59_col5, #T_a0baa_row60_col0, #T_a0baa_row60_col1, #T_a0baa_row60_col2, #T_a0baa_row60_col3, #T_a0baa_row60_col4, #T_a0baa_row60_col5, #T_a0baa_row61_col0, #T_a0baa_row61_col1, #T_a0baa_row61_col2, #T_a0baa_row61_col3, #T_a0baa_row61_col4, #T_a0baa_row61_col5, #T_a0baa_row62_col0, #T_a0baa_row62_col1, #T_a0baa_row62_col2, #T_a0baa_row62_col3, #T_a0baa_row62_col4, #T_a0baa_row62_col5, #T_a0baa_row63_col0, #T_a0baa_row63_col1, #T_a0baa_row63_col2, #T_a0baa_row63_col3, #T_a0baa_row63_col4, #T_a0baa_row63_col5, #T_a0baa_row64_col0, #T_a0baa_row64_col1, #T_a0baa_row64_col2, #T_a0baa_row64_col3, #T_a0baa_row64_col4, #T_a0baa_row64_col5, #T_a0baa_row65_col0, #T_a0baa_row65_col1, #T_a0baa_row65_col2, #T_a0baa_row65_col3, #T_a0baa_row65_col4, #T_a0baa_row65_col5 {
  text-align: left;
}
</style>
<table id="T_a0baa">
  <thead>
  </thead>
  <tbody>
    <tr>
      <td id="T_a0baa_row0_col0" class="data row0 col0" >  0  IDfg</td>
      <td id="T_a0baa_row0_col1" class="data row0 col1" >  1  Season</td>
      <td id="T_a0baa_row0_col2" class="data row0 col2" >  2  Name</td>
      <td id="T_a0baa_row0_col3" class="data row0 col3" >  3  Team</td>
      <td id="T_a0baa_row0_col4" class="data row0 col4" >  4  Age</td>
      <td id="T_a0baa_row0_col5" class="data row0 col5" >  5  W</td>
    </tr>
    <tr>
      <td id="T_a0baa_row1_col0" class="data row1 col0" >  6  L</td>
      <td id="T_a0baa_row1_col1" class="data row1 col1" >  7  WAR</td>
      <td id="T_a0baa_row1_col2" class="data row1 col2" >  8  ERA</td>
      <td id="T_a0baa_row1_col3" class="data row1 col3" >  9  G</td>
      <td id="T_a0baa_row1_col4" class="data row1 col4" > 10  GS</td>
      <td id="T_a0baa_row1_col5" class="data row1 col5" > 11  CG</td>
    </tr>
    <tr>
      <td id="T_a0baa_row2_col0" class="data row2 col0" > 12  ShO</td>
      <td id="T_a0baa_row2_col1" class="data row2 col1" > 13  SV</td>
      <td id="T_a0baa_row2_col2" class="data row2 col2" > 14  BS</td>
      <td id="T_a0baa_row2_col3" class="data row2 col3" > 15  IP</td>
      <td id="T_a0baa_row2_col4" class="data row2 col4" > 16  TBF</td>
      <td id="T_a0baa_row2_col5" class="data row2 col5" > 17  H</td>
    </tr>
    <tr>
      <td id="T_a0baa_row3_col0" class="data row3 col0" > 18  R</td>
      <td id="T_a0baa_row3_col1" class="data row3 col1" > 19  ER</td>
      <td id="T_a0baa_row3_col2" class="data row3 col2" > 20  HR</td>
      <td id="T_a0baa_row3_col3" class="data row3 col3" > 21  BB</td>
      <td id="T_a0baa_row3_col4" class="data row3 col4" > 22  IBB</td>
      <td id="T_a0baa_row3_col5" class="data row3 col5" > 23  HBP</td>
    </tr>
    <tr>
      <td id="T_a0baa_row4_col0" class="data row4 col0" > 24  WP</td>
      <td id="T_a0baa_row4_col1" class="data row4 col1" > 25  BK</td>
      <td id="T_a0baa_row4_col2" class="data row4 col2" > 26  SO</td>
      <td id="T_a0baa_row4_col3" class="data row4 col3" > 27  GB</td>
      <td id="T_a0baa_row4_col4" class="data row4 col4" > 28  FB</td>
      <td id="T_a0baa_row4_col5" class="data row4 col5" > 29  LD</td>
    </tr>
    <tr>
      <td id="T_a0baa_row5_col0" class="data row5 col0" > 30  IFFB</td>
      <td id="T_a0baa_row5_col1" class="data row5 col1" > 31  Balls</td>
      <td id="T_a0baa_row5_col2" class="data row5 col2" > 32  Strikes</td>
      <td id="T_a0baa_row5_col3" class="data row5 col3" > 33  Pitches</td>
      <td id="T_a0baa_row5_col4" class="data row5 col4" > 34  RS</td>
      <td id="T_a0baa_row5_col5" class="data row5 col5" > 35  IFH</td>
    </tr>
    <tr>
      <td id="T_a0baa_row6_col0" class="data row6 col0" > 36  BU</td>
      <td id="T_a0baa_row6_col1" class="data row6 col1" > 37  BUH</td>
      <td id="T_a0baa_row6_col2" class="data row6 col2" > 38  K/9</td>
      <td id="T_a0baa_row6_col3" class="data row6 col3" > 39  BB/9</td>
      <td id="T_a0baa_row6_col4" class="data row6 col4" > 40  K/BB</td>
      <td id="T_a0baa_row6_col5" class="data row6 col5" > 41  H/9</td>
    </tr>
    <tr>
      <td id="T_a0baa_row7_col0" class="data row7 col0" > 42  HR/9</td>
      <td id="T_a0baa_row7_col1" class="data row7 col1" > 43  AVG</td>
      <td id="T_a0baa_row7_col2" class="data row7 col2" > 44  WHIP</td>
      <td id="T_a0baa_row7_col3" class="data row7 col3" > 45  BABIP</td>
      <td id="T_a0baa_row7_col4" class="data row7 col4" > 46  LOB%</td>
      <td id="T_a0baa_row7_col5" class="data row7 col5" > 47  FIP</td>
    </tr>
    <tr>
      <td id="T_a0baa_row8_col0" class="data row8 col0" > 48  GB/FB</td>
      <td id="T_a0baa_row8_col1" class="data row8 col1" > 49  LD%</td>
      <td id="T_a0baa_row8_col2" class="data row8 col2" > 50  GB%</td>
      <td id="T_a0baa_row8_col3" class="data row8 col3" > 51  FB%</td>
      <td id="T_a0baa_row8_col4" class="data row8 col4" > 52  IFFB%</td>
      <td id="T_a0baa_row8_col5" class="data row8 col5" > 53  HR/FB</td>
    </tr>
    <tr>
      <td id="T_a0baa_row9_col0" class="data row9 col0" > 54  IFH%</td>
      <td id="T_a0baa_row9_col1" class="data row9 col1" > 55  BUH%</td>
      <td id="T_a0baa_row9_col2" class="data row9 col2" > 56  Starting</td>
      <td id="T_a0baa_row9_col3" class="data row9 col3" > 57  Start-IP</td>
      <td id="T_a0baa_row9_col4" class="data row9 col4" > 58  Relieving</td>
      <td id="T_a0baa_row9_col5" class="data row9 col5" > 59  Relief-IP</td>
    </tr>
    <tr>
      <td id="T_a0baa_row10_col0" class="data row10 col0" > 60  RAR</td>
      <td id="T_a0baa_row10_col1" class="data row10 col1" > 61  Dollars</td>
      <td id="T_a0baa_row10_col2" class="data row10 col2" > 62  tERA</td>
      <td id="T_a0baa_row10_col3" class="data row10 col3" > 63  xFIP</td>
      <td id="T_a0baa_row10_col4" class="data row10 col4" > 64  WPA</td>
      <td id="T_a0baa_row10_col5" class="data row10 col5" > 65  -WPA</td>
    </tr>
    <tr>
      <td id="T_a0baa_row11_col0" class="data row11 col0" > 66  +WPA</td>
      <td id="T_a0baa_row11_col1" class="data row11 col1" > 67  RE24</td>
      <td id="T_a0baa_row11_col2" class="data row11 col2" > 68  REW</td>
      <td id="T_a0baa_row11_col3" class="data row11 col3" > 69  pLI</td>
      <td id="T_a0baa_row11_col4" class="data row11 col4" > 70  inLI</td>
      <td id="T_a0baa_row11_col5" class="data row11 col5" > 71  gmLI</td>
    </tr>
    <tr>
      <td id="T_a0baa_row12_col0" class="data row12 col0" > 72  exLI</td>
      <td id="T_a0baa_row12_col1" class="data row12 col1" > 73  Pulls</td>
      <td id="T_a0baa_row12_col2" class="data row12 col2" > 74  WPA/LI</td>
      <td id="T_a0baa_row12_col3" class="data row12 col3" > 75  Clutch</td>
      <td id="T_a0baa_row12_col4" class="data row12 col4" > 76  FB% 2</td>
      <td id="T_a0baa_row12_col5" class="data row12 col5" > 77  FBv</td>
    </tr>
    <tr>
      <td id="T_a0baa_row13_col0" class="data row13 col0" > 78  SL%</td>
      <td id="T_a0baa_row13_col1" class="data row13 col1" > 79  SLv</td>
      <td id="T_a0baa_row13_col2" class="data row13 col2" > 80  CT%</td>
      <td id="T_a0baa_row13_col3" class="data row13 col3" > 81  CTv</td>
      <td id="T_a0baa_row13_col4" class="data row13 col4" > 82  CB%</td>
      <td id="T_a0baa_row13_col5" class="data row13 col5" > 83  CBv</td>
    </tr>
    <tr>
      <td id="T_a0baa_row14_col0" class="data row14 col0" > 84  CH%</td>
      <td id="T_a0baa_row14_col1" class="data row14 col1" > 85  CHv</td>
      <td id="T_a0baa_row14_col2" class="data row14 col2" > 86  SF%</td>
      <td id="T_a0baa_row14_col3" class="data row14 col3" > 87  SFv</td>
      <td id="T_a0baa_row14_col4" class="data row14 col4" > 88  KN%</td>
      <td id="T_a0baa_row14_col5" class="data row14 col5" > 89  KNv</td>
    </tr>
    <tr>
      <td id="T_a0baa_row15_col0" class="data row15 col0" > 90  XX%</td>
      <td id="T_a0baa_row15_col1" class="data row15 col1" > 91  PO%</td>
      <td id="T_a0baa_row15_col2" class="data row15 col2" > 92  wFB</td>
      <td id="T_a0baa_row15_col3" class="data row15 col3" > 93  wSL</td>
      <td id="T_a0baa_row15_col4" class="data row15 col4" > 94  wCT</td>
      <td id="T_a0baa_row15_col5" class="data row15 col5" > 95  wCB</td>
    </tr>
    <tr>
      <td id="T_a0baa_row16_col0" class="data row16 col0" > 96  wCH</td>
      <td id="T_a0baa_row16_col1" class="data row16 col1" > 97  wSF</td>
      <td id="T_a0baa_row16_col2" class="data row16 col2" > 98  wKN</td>
      <td id="T_a0baa_row16_col3" class="data row16 col3" > 99  wFB/C</td>
      <td id="T_a0baa_row16_col4" class="data row16 col4" >100  wSL/C</td>
      <td id="T_a0baa_row16_col5" class="data row16 col5" >101  wCT/C</td>
    </tr>
    <tr>
      <td id="T_a0baa_row17_col0" class="data row17 col0" >102  wCB/C</td>
      <td id="T_a0baa_row17_col1" class="data row17 col1" >103  wCH/C</td>
      <td id="T_a0baa_row17_col2" class="data row17 col2" >104  wSF/C</td>
      <td id="T_a0baa_row17_col3" class="data row17 col3" >105  wKN/C</td>
      <td id="T_a0baa_row17_col4" class="data row17 col4" >106  O-Swing%</td>
      <td id="T_a0baa_row17_col5" class="data row17 col5" >107  Z-Swing%</td>
    </tr>
    <tr>
      <td id="T_a0baa_row18_col0" class="data row18 col0" >108  Swing%</td>
      <td id="T_a0baa_row18_col1" class="data row18 col1" >109  O-Contact%</td>
      <td id="T_a0baa_row18_col2" class="data row18 col2" >110  Z-Contact%</td>
      <td id="T_a0baa_row18_col3" class="data row18 col3" >111  Contact%</td>
      <td id="T_a0baa_row18_col4" class="data row18 col4" >112  Zone%</td>
      <td id="T_a0baa_row18_col5" class="data row18 col5" >113  F-Strike%</td>
    </tr>
    <tr>
      <td id="T_a0baa_row19_col0" class="data row19 col0" >114  SwStr%</td>
      <td id="T_a0baa_row19_col1" class="data row19 col1" >115  HLD</td>
      <td id="T_a0baa_row19_col2" class="data row19 col2" >116  SD</td>
      <td id="T_a0baa_row19_col3" class="data row19 col3" >117  MD</td>
      <td id="T_a0baa_row19_col4" class="data row19 col4" >118  ERA-</td>
      <td id="T_a0baa_row19_col5" class="data row19 col5" >119  FIP-</td>
    </tr>
    <tr>
      <td id="T_a0baa_row20_col0" class="data row20 col0" >120  xFIP-</td>
      <td id="T_a0baa_row20_col1" class="data row20 col1" >121  K%</td>
      <td id="T_a0baa_row20_col2" class="data row20 col2" >122  BB%</td>
      <td id="T_a0baa_row20_col3" class="data row20 col3" >123  SIERA</td>
      <td id="T_a0baa_row20_col4" class="data row20 col4" >124  RS/9</td>
      <td id="T_a0baa_row20_col5" class="data row20 col5" >125  E-F</td>
    </tr>
    <tr>
      <td id="T_a0baa_row21_col0" class="data row21 col0" >126  FA% (sc)</td>
      <td id="T_a0baa_row21_col1" class="data row21 col1" >127  FT% (sc)</td>
      <td id="T_a0baa_row21_col2" class="data row21 col2" >128  FC% (sc)</td>
      <td id="T_a0baa_row21_col3" class="data row21 col3" >129  FS% (sc)</td>
      <td id="T_a0baa_row21_col4" class="data row21 col4" >130  FO% (sc)</td>
      <td id="T_a0baa_row21_col5" class="data row21 col5" >131  SI% (sc)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row22_col0" class="data row22 col0" >132  SL% (sc)</td>
      <td id="T_a0baa_row22_col1" class="data row22 col1" >133  CU% (sc)</td>
      <td id="T_a0baa_row22_col2" class="data row22 col2" >134  KC% (sc)</td>
      <td id="T_a0baa_row22_col3" class="data row22 col3" >135  EP% (sc)</td>
      <td id="T_a0baa_row22_col4" class="data row22 col4" >136  CH% (sc)</td>
      <td id="T_a0baa_row22_col5" class="data row22 col5" >137  SC% (sc)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row23_col0" class="data row23 col0" >138  KN% (sc)</td>
      <td id="T_a0baa_row23_col1" class="data row23 col1" >139  UN% (sc)</td>
      <td id="T_a0baa_row23_col2" class="data row23 col2" >140  vFA (sc)</td>
      <td id="T_a0baa_row23_col3" class="data row23 col3" >141  vFT (sc)</td>
      <td id="T_a0baa_row23_col4" class="data row23 col4" >142  vFC (sc)</td>
      <td id="T_a0baa_row23_col5" class="data row23 col5" >143  vFS (sc)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row24_col0" class="data row24 col0" >144  vFO (sc)</td>
      <td id="T_a0baa_row24_col1" class="data row24 col1" >145  vSI (sc)</td>
      <td id="T_a0baa_row24_col2" class="data row24 col2" >146  vSL (sc)</td>
      <td id="T_a0baa_row24_col3" class="data row24 col3" >147  vCU (sc)</td>
      <td id="T_a0baa_row24_col4" class="data row24 col4" >148  vKC (sc)</td>
      <td id="T_a0baa_row24_col5" class="data row24 col5" >149  vEP (sc)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row25_col0" class="data row25 col0" >150  vCH (sc)</td>
      <td id="T_a0baa_row25_col1" class="data row25 col1" >151  vSC (sc)</td>
      <td id="T_a0baa_row25_col2" class="data row25 col2" >152  vKN (sc)</td>
      <td id="T_a0baa_row25_col3" class="data row25 col3" >153  FA-X (sc)</td>
      <td id="T_a0baa_row25_col4" class="data row25 col4" >154  FT-X (sc)</td>
      <td id="T_a0baa_row25_col5" class="data row25 col5" >155  FC-X (sc)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row26_col0" class="data row26 col0" >156  FS-X (sc)</td>
      <td id="T_a0baa_row26_col1" class="data row26 col1" >157  FO-X (sc)</td>
      <td id="T_a0baa_row26_col2" class="data row26 col2" >158  SI-X (sc)</td>
      <td id="T_a0baa_row26_col3" class="data row26 col3" >159  SL-X (sc)</td>
      <td id="T_a0baa_row26_col4" class="data row26 col4" >160  CU-X (sc)</td>
      <td id="T_a0baa_row26_col5" class="data row26 col5" >161  KC-X (sc)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row27_col0" class="data row27 col0" >162  EP-X (sc)</td>
      <td id="T_a0baa_row27_col1" class="data row27 col1" >163  CH-X (sc)</td>
      <td id="T_a0baa_row27_col2" class="data row27 col2" >164  SC-X (sc)</td>
      <td id="T_a0baa_row27_col3" class="data row27 col3" >165  KN-X (sc)</td>
      <td id="T_a0baa_row27_col4" class="data row27 col4" >166  FA-Z (sc)</td>
      <td id="T_a0baa_row27_col5" class="data row27 col5" >167  FT-Z (sc)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row28_col0" class="data row28 col0" >168  FC-Z (sc)</td>
      <td id="T_a0baa_row28_col1" class="data row28 col1" >169  FS-Z (sc)</td>
      <td id="T_a0baa_row28_col2" class="data row28 col2" >170  FO-Z (sc)</td>
      <td id="T_a0baa_row28_col3" class="data row28 col3" >171  SI-Z (sc)</td>
      <td id="T_a0baa_row28_col4" class="data row28 col4" >172  SL-Z (sc)</td>
      <td id="T_a0baa_row28_col5" class="data row28 col5" >173  CU-Z (sc)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row29_col0" class="data row29 col0" >174  KC-Z (sc)</td>
      <td id="T_a0baa_row29_col1" class="data row29 col1" >175  EP-Z (sc)</td>
      <td id="T_a0baa_row29_col2" class="data row29 col2" >176  CH-Z (sc)</td>
      <td id="T_a0baa_row29_col3" class="data row29 col3" >177  SC-Z (sc)</td>
      <td id="T_a0baa_row29_col4" class="data row29 col4" >178  KN-Z (sc)</td>
      <td id="T_a0baa_row29_col5" class="data row29 col5" >179  wFA (sc)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row30_col0" class="data row30 col0" >180  wFT (sc)</td>
      <td id="T_a0baa_row30_col1" class="data row30 col1" >181  wFC (sc)</td>
      <td id="T_a0baa_row30_col2" class="data row30 col2" >182  wFS (sc)</td>
      <td id="T_a0baa_row30_col3" class="data row30 col3" >183  wFO (sc)</td>
      <td id="T_a0baa_row30_col4" class="data row30 col4" >184  wSI (sc)</td>
      <td id="T_a0baa_row30_col5" class="data row30 col5" >185  wSL (sc)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row31_col0" class="data row31 col0" >186  wCU (sc)</td>
      <td id="T_a0baa_row31_col1" class="data row31 col1" >187  wKC (sc)</td>
      <td id="T_a0baa_row31_col2" class="data row31 col2" >188  wEP (sc)</td>
      <td id="T_a0baa_row31_col3" class="data row31 col3" >189  wCH (sc)</td>
      <td id="T_a0baa_row31_col4" class="data row31 col4" >190  wSC (sc)</td>
      <td id="T_a0baa_row31_col5" class="data row31 col5" >191  wKN (sc)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row32_col0" class="data row32 col0" >192  wFA/C (sc)</td>
      <td id="T_a0baa_row32_col1" class="data row32 col1" >193  wFT/C (sc)</td>
      <td id="T_a0baa_row32_col2" class="data row32 col2" >194  wFC/C (sc)</td>
      <td id="T_a0baa_row32_col3" class="data row32 col3" >195  wFS/C (sc)</td>
      <td id="T_a0baa_row32_col4" class="data row32 col4" >196  wFO/C (sc)</td>
      <td id="T_a0baa_row32_col5" class="data row32 col5" >197  wSI/C (sc)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row33_col0" class="data row33 col0" >198  wSL/C (sc)</td>
      <td id="T_a0baa_row33_col1" class="data row33 col1" >199  wCU/C (sc)</td>
      <td id="T_a0baa_row33_col2" class="data row33 col2" >200  wKC/C (sc)</td>
      <td id="T_a0baa_row33_col3" class="data row33 col3" >201  wEP/C (sc)</td>
      <td id="T_a0baa_row33_col4" class="data row33 col4" >202  wCH/C (sc)</td>
      <td id="T_a0baa_row33_col5" class="data row33 col5" >203  wSC/C (sc)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row34_col0" class="data row34 col0" >204  wKN/C (sc)</td>
      <td id="T_a0baa_row34_col1" class="data row34 col1" >205  O-Swing% (sc)</td>
      <td id="T_a0baa_row34_col2" class="data row34 col2" >206  Z-Swing% (sc)</td>
      <td id="T_a0baa_row34_col3" class="data row34 col3" >207  Swing% (sc)</td>
      <td id="T_a0baa_row34_col4" class="data row34 col4" >208  O-Contact% (sc)</td>
      <td id="T_a0baa_row34_col5" class="data row34 col5" >209  Z-Contact% (sc)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row35_col0" class="data row35 col0" >210  Contact% (sc)</td>
      <td id="T_a0baa_row35_col1" class="data row35 col1" >211  Zone% (sc)</td>
      <td id="T_a0baa_row35_col2" class="data row35 col2" >212  Pace</td>
      <td id="T_a0baa_row35_col3" class="data row35 col3" >213  RA9-WAR</td>
      <td id="T_a0baa_row35_col4" class="data row35 col4" >214  BIP-Wins</td>
      <td id="T_a0baa_row35_col5" class="data row35 col5" >215  LOB-Wins</td>
    </tr>
    <tr>
      <td id="T_a0baa_row36_col0" class="data row36 col0" >216  FDP-Wins</td>
      <td id="T_a0baa_row36_col1" class="data row36 col1" >217  Age Rng</td>
      <td id="T_a0baa_row36_col2" class="data row36 col2" >218  K-BB%</td>
      <td id="T_a0baa_row36_col3" class="data row36 col3" >219  Pull%</td>
      <td id="T_a0baa_row36_col4" class="data row36 col4" >220  Cent%</td>
      <td id="T_a0baa_row36_col5" class="data row36 col5" >221  Oppo%</td>
    </tr>
    <tr>
      <td id="T_a0baa_row37_col0" class="data row37 col0" >222  Soft%</td>
      <td id="T_a0baa_row37_col1" class="data row37 col1" >223  Med%</td>
      <td id="T_a0baa_row37_col2" class="data row37 col2" >224  Hard%</td>
      <td id="T_a0baa_row37_col3" class="data row37 col3" >225  kwERA</td>
      <td id="T_a0baa_row37_col4" class="data row37 col4" >226  TTO%</td>
      <td id="T_a0baa_row37_col5" class="data row37 col5" >227  CH% (pi)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row38_col0" class="data row38 col0" >228  CS% (pi)</td>
      <td id="T_a0baa_row38_col1" class="data row38 col1" >229  CU% (pi)</td>
      <td id="T_a0baa_row38_col2" class="data row38 col2" >230  FA% (pi)</td>
      <td id="T_a0baa_row38_col3" class="data row38 col3" >231  FC% (pi)</td>
      <td id="T_a0baa_row38_col4" class="data row38 col4" >232  FS% (pi)</td>
      <td id="T_a0baa_row38_col5" class="data row38 col5" >233  KN% (pi)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row39_col0" class="data row39 col0" >234  SB% (pi)</td>
      <td id="T_a0baa_row39_col1" class="data row39 col1" >235  SI% (pi)</td>
      <td id="T_a0baa_row39_col2" class="data row39 col2" >236  SL% (pi)</td>
      <td id="T_a0baa_row39_col3" class="data row39 col3" >237  XX% (pi)</td>
      <td id="T_a0baa_row39_col4" class="data row39 col4" >238  vCH (pi)</td>
      <td id="T_a0baa_row39_col5" class="data row39 col5" >239  vCS (pi)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row40_col0" class="data row40 col0" >240  vCU (pi)</td>
      <td id="T_a0baa_row40_col1" class="data row40 col1" >241  vFA (pi)</td>
      <td id="T_a0baa_row40_col2" class="data row40 col2" >242  vFC (pi)</td>
      <td id="T_a0baa_row40_col3" class="data row40 col3" >243  vFS (pi)</td>
      <td id="T_a0baa_row40_col4" class="data row40 col4" >244  vKN (pi)</td>
      <td id="T_a0baa_row40_col5" class="data row40 col5" >245  vSB (pi)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row41_col0" class="data row41 col0" >246  vSI (pi)</td>
      <td id="T_a0baa_row41_col1" class="data row41 col1" >247  vSL (pi)</td>
      <td id="T_a0baa_row41_col2" class="data row41 col2" >248  vXX (pi)</td>
      <td id="T_a0baa_row41_col3" class="data row41 col3" >249  CH-X (pi)</td>
      <td id="T_a0baa_row41_col4" class="data row41 col4" >250  CS-X (pi)</td>
      <td id="T_a0baa_row41_col5" class="data row41 col5" >251  CU-X (pi)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row42_col0" class="data row42 col0" >252  FA-X (pi)</td>
      <td id="T_a0baa_row42_col1" class="data row42 col1" >253  FC-X (pi)</td>
      <td id="T_a0baa_row42_col2" class="data row42 col2" >254  FS-X (pi)</td>
      <td id="T_a0baa_row42_col3" class="data row42 col3" >255  KN-X (pi)</td>
      <td id="T_a0baa_row42_col4" class="data row42 col4" >256  SB-X (pi)</td>
      <td id="T_a0baa_row42_col5" class="data row42 col5" >257  SI-X (pi)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row43_col0" class="data row43 col0" >258  SL-X (pi)</td>
      <td id="T_a0baa_row43_col1" class="data row43 col1" >259  XX-X (pi)</td>
      <td id="T_a0baa_row43_col2" class="data row43 col2" >260  CH-Z (pi)</td>
      <td id="T_a0baa_row43_col3" class="data row43 col3" >261  CS-Z (pi)</td>
      <td id="T_a0baa_row43_col4" class="data row43 col4" >262  CU-Z (pi)</td>
      <td id="T_a0baa_row43_col5" class="data row43 col5" >263  FA-Z (pi)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row44_col0" class="data row44 col0" >264  FC-Z (pi)</td>
      <td id="T_a0baa_row44_col1" class="data row44 col1" >265  FS-Z (pi)</td>
      <td id="T_a0baa_row44_col2" class="data row44 col2" >266  KN-Z (pi)</td>
      <td id="T_a0baa_row44_col3" class="data row44 col3" >267  SB-Z (pi)</td>
      <td id="T_a0baa_row44_col4" class="data row44 col4" >268  SI-Z (pi)</td>
      <td id="T_a0baa_row44_col5" class="data row44 col5" >269  SL-Z (pi)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row45_col0" class="data row45 col0" >270  XX-Z (pi)</td>
      <td id="T_a0baa_row45_col1" class="data row45 col1" >271  wCH (pi)</td>
      <td id="T_a0baa_row45_col2" class="data row45 col2" >272  wCS (pi)</td>
      <td id="T_a0baa_row45_col3" class="data row45 col3" >273  wCU (pi)</td>
      <td id="T_a0baa_row45_col4" class="data row45 col4" >274  wFA (pi)</td>
      <td id="T_a0baa_row45_col5" class="data row45 col5" >275  wFC (pi)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row46_col0" class="data row46 col0" >276  wFS (pi)</td>
      <td id="T_a0baa_row46_col1" class="data row46 col1" >277  wKN (pi)</td>
      <td id="T_a0baa_row46_col2" class="data row46 col2" >278  wSB (pi)</td>
      <td id="T_a0baa_row46_col3" class="data row46 col3" >279  wSI (pi)</td>
      <td id="T_a0baa_row46_col4" class="data row46 col4" >280  wSL (pi)</td>
      <td id="T_a0baa_row46_col5" class="data row46 col5" >281  wXX (pi)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row47_col0" class="data row47 col0" >282  wCH/C (pi)</td>
      <td id="T_a0baa_row47_col1" class="data row47 col1" >283  wCS/C (pi)</td>
      <td id="T_a0baa_row47_col2" class="data row47 col2" >284  wCU/C (pi)</td>
      <td id="T_a0baa_row47_col3" class="data row47 col3" >285  wFA/C (pi)</td>
      <td id="T_a0baa_row47_col4" class="data row47 col4" >286  wFC/C (pi)</td>
      <td id="T_a0baa_row47_col5" class="data row47 col5" >287  wFS/C (pi)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row48_col0" class="data row48 col0" >288  wKN/C (pi)</td>
      <td id="T_a0baa_row48_col1" class="data row48 col1" >289  wSB/C (pi)</td>
      <td id="T_a0baa_row48_col2" class="data row48 col2" >290  wSI/C (pi)</td>
      <td id="T_a0baa_row48_col3" class="data row48 col3" >291  wSL/C (pi)</td>
      <td id="T_a0baa_row48_col4" class="data row48 col4" >292  wXX/C (pi)</td>
      <td id="T_a0baa_row48_col5" class="data row48 col5" >293  O-Swing% (pi)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row49_col0" class="data row49 col0" >294  Z-Swing% (pi)</td>
      <td id="T_a0baa_row49_col1" class="data row49 col1" >295  Swing% (pi)</td>
      <td id="T_a0baa_row49_col2" class="data row49 col2" >296  O-Contact% (pi)</td>
      <td id="T_a0baa_row49_col3" class="data row49 col3" >297  Z-Contact% (pi)</td>
      <td id="T_a0baa_row49_col4" class="data row49 col4" >298  Contact% (pi)</td>
      <td id="T_a0baa_row49_col5" class="data row49 col5" >299  Zone% (pi)</td>
    </tr>
    <tr>
      <td id="T_a0baa_row50_col0" class="data row50 col0" >300  Pace (pi)</td>
      <td id="T_a0baa_row50_col1" class="data row50 col1" >301  FRM</td>
      <td id="T_a0baa_row50_col2" class="data row50 col2" >302  K/9+</td>
      <td id="T_a0baa_row50_col3" class="data row50 col3" >303  BB/9+</td>
      <td id="T_a0baa_row50_col4" class="data row50 col4" >304  K/BB+</td>
      <td id="T_a0baa_row50_col5" class="data row50 col5" >305  H/9+</td>
    </tr>
    <tr>
      <td id="T_a0baa_row51_col0" class="data row51 col0" >306  HR/9+</td>
      <td id="T_a0baa_row51_col1" class="data row51 col1" >307  AVG+</td>
      <td id="T_a0baa_row51_col2" class="data row51 col2" >308  WHIP+</td>
      <td id="T_a0baa_row51_col3" class="data row51 col3" >309  BABIP+</td>
      <td id="T_a0baa_row51_col4" class="data row51 col4" >310  LOB%+</td>
      <td id="T_a0baa_row51_col5" class="data row51 col5" >311  K%+</td>
    </tr>
    <tr>
      <td id="T_a0baa_row52_col0" class="data row52 col0" >312  BB%+</td>
      <td id="T_a0baa_row52_col1" class="data row52 col1" >313  LD%+</td>
      <td id="T_a0baa_row52_col2" class="data row52 col2" >314  GB%+</td>
      <td id="T_a0baa_row52_col3" class="data row52 col3" >315  FB%+</td>
      <td id="T_a0baa_row52_col4" class="data row52 col4" >316  HR/FB%+</td>
      <td id="T_a0baa_row52_col5" class="data row52 col5" >317  Pull%+</td>
    </tr>
    <tr>
      <td id="T_a0baa_row53_col0" class="data row53 col0" >318  Cent%+</td>
      <td id="T_a0baa_row53_col1" class="data row53 col1" >319  Oppo%+</td>
      <td id="T_a0baa_row53_col2" class="data row53 col2" >320  Soft%+</td>
      <td id="T_a0baa_row53_col3" class="data row53 col3" >321  Med%+</td>
      <td id="T_a0baa_row53_col4" class="data row53 col4" >322  Hard%+</td>
      <td id="T_a0baa_row53_col5" class="data row53 col5" >323  EV</td>
    </tr>
    <tr>
      <td id="T_a0baa_row54_col0" class="data row54 col0" >324  LA</td>
      <td id="T_a0baa_row54_col1" class="data row54 col1" >325  Barrels</td>
      <td id="T_a0baa_row54_col2" class="data row54 col2" >326  Barrel%</td>
      <td id="T_a0baa_row54_col3" class="data row54 col3" >327  maxEV</td>
      <td id="T_a0baa_row54_col4" class="data row54 col4" >328  HardHit</td>
      <td id="T_a0baa_row54_col5" class="data row54 col5" >329  HardHit%</td>
    </tr>
    <tr>
      <td id="T_a0baa_row55_col0" class="data row55 col0" >330  Events</td>
      <td id="T_a0baa_row55_col1" class="data row55 col1" >331  CStr%</td>
      <td id="T_a0baa_row55_col2" class="data row55 col2" >332  CSW%</td>
      <td id="T_a0baa_row55_col3" class="data row55 col3" >333  xERA</td>
      <td id="T_a0baa_row55_col4" class="data row55 col4" >334  botERA</td>
      <td id="T_a0baa_row55_col5" class="data row55 col5" >335  botOvr CH</td>
    </tr>
    <tr>
      <td id="T_a0baa_row56_col0" class="data row56 col0" >336  botStf CH</td>
      <td id="T_a0baa_row56_col1" class="data row56 col1" >337  botCmd CH</td>
      <td id="T_a0baa_row56_col2" class="data row56 col2" >338  botOvr CU</td>
      <td id="T_a0baa_row56_col3" class="data row56 col3" >339  botStf CU</td>
      <td id="T_a0baa_row56_col4" class="data row56 col4" >340  botCmd CU</td>
      <td id="T_a0baa_row56_col5" class="data row56 col5" >341  botOvr FA</td>
    </tr>
    <tr>
      <td id="T_a0baa_row57_col0" class="data row57 col0" >342  botStf FA</td>
      <td id="T_a0baa_row57_col1" class="data row57 col1" >343  botCmd FA</td>
      <td id="T_a0baa_row57_col2" class="data row57 col2" >344  botOvr SI</td>
      <td id="T_a0baa_row57_col3" class="data row57 col3" >345  botStf SI</td>
      <td id="T_a0baa_row57_col4" class="data row57 col4" >346  botCmd SI</td>
      <td id="T_a0baa_row57_col5" class="data row57 col5" >347  botOvr SL</td>
    </tr>
    <tr>
      <td id="T_a0baa_row58_col0" class="data row58 col0" >348  botStf SL</td>
      <td id="T_a0baa_row58_col1" class="data row58 col1" >349  botCmd SL</td>
      <td id="T_a0baa_row58_col2" class="data row58 col2" >350  botOvr KC</td>
      <td id="T_a0baa_row58_col3" class="data row58 col3" >351  botStf KC</td>
      <td id="T_a0baa_row58_col4" class="data row58 col4" >352  botCmd KC</td>
      <td id="T_a0baa_row58_col5" class="data row58 col5" >353  botOvr FC</td>
    </tr>
    <tr>
      <td id="T_a0baa_row59_col0" class="data row59 col0" >354  botStf FC</td>
      <td id="T_a0baa_row59_col1" class="data row59 col1" >355  botCmd FC</td>
      <td id="T_a0baa_row59_col2" class="data row59 col2" >356  botOvr FS</td>
      <td id="T_a0baa_row59_col3" class="data row59 col3" >357  botStf FS</td>
      <td id="T_a0baa_row59_col4" class="data row59 col4" >358  botCmd FS</td>
      <td id="T_a0baa_row59_col5" class="data row59 col5" >359  botOvr</td>
    </tr>
    <tr>
      <td id="T_a0baa_row60_col0" class="data row60 col0" >360  botStf</td>
      <td id="T_a0baa_row60_col1" class="data row60 col1" >361  botCmd</td>
      <td id="T_a0baa_row60_col2" class="data row60 col2" >362  botxRV100</td>
      <td id="T_a0baa_row60_col3" class="data row60 col3" >363  Stf+ CH</td>
      <td id="T_a0baa_row60_col4" class="data row60 col4" >364  Loc+ CH</td>
      <td id="T_a0baa_row60_col5" class="data row60 col5" >365  Pit+ CH</td>
    </tr>
    <tr>
      <td id="T_a0baa_row61_col0" class="data row61 col0" >366  Stf+ CU</td>
      <td id="T_a0baa_row61_col1" class="data row61 col1" >367  Loc+ CU</td>
      <td id="T_a0baa_row61_col2" class="data row61 col2" >368  Pit+ CU</td>
      <td id="T_a0baa_row61_col3" class="data row61 col3" >369  Stf+ FA</td>
      <td id="T_a0baa_row61_col4" class="data row61 col4" >370  Loc+ FA</td>
      <td id="T_a0baa_row61_col5" class="data row61 col5" >371  Pit+ FA</td>
    </tr>
    <tr>
      <td id="T_a0baa_row62_col0" class="data row62 col0" >372  Stf+ SI</td>
      <td id="T_a0baa_row62_col1" class="data row62 col1" >373  Loc+ SI</td>
      <td id="T_a0baa_row62_col2" class="data row62 col2" >374  Pit+ SI</td>
      <td id="T_a0baa_row62_col3" class="data row62 col3" >375  Stf+ SL</td>
      <td id="T_a0baa_row62_col4" class="data row62 col4" >376  Loc+ SL</td>
      <td id="T_a0baa_row62_col5" class="data row62 col5" >377  Pit+ SL</td>
    </tr>
    <tr>
      <td id="T_a0baa_row63_col0" class="data row63 col0" >378  Stf+ KC</td>
      <td id="T_a0baa_row63_col1" class="data row63 col1" >379  Loc+ KC</td>
      <td id="T_a0baa_row63_col2" class="data row63 col2" >380  Pit+ KC</td>
      <td id="T_a0baa_row63_col3" class="data row63 col3" >381  Stf+ FC</td>
      <td id="T_a0baa_row63_col4" class="data row63 col4" >382  Loc+ FC</td>
      <td id="T_a0baa_row63_col5" class="data row63 col5" >383  Pit+ FC</td>
    </tr>
    <tr>
      <td id="T_a0baa_row64_col0" class="data row64 col0" >384  Stf+ FS</td>
      <td id="T_a0baa_row64_col1" class="data row64 col1" >385  Loc+ FS</td>
      <td id="T_a0baa_row64_col2" class="data row64 col2" >386  Pit+ FS</td>
      <td id="T_a0baa_row64_col3" class="data row64 col3" >387  Stuff+</td>
      <td id="T_a0baa_row64_col4" class="data row64 col4" >388  Location+</td>
      <td id="T_a0baa_row64_col5" class="data row64 col5" >389  Pitching+</td>
    </tr>
    <tr>
      <td id="T_a0baa_row65_col0" class="data row65 col0" >390  Stf+ FO</td>
      <td id="T_a0baa_row65_col1" class="data row65 col1" >391  Loc+ FO</td>
      <td id="T_a0baa_row65_col2" class="data row65 col2" >392  Pit+ FO</td>
      <td id="T_a0baa_row65_col3" class="data row65 col3" ></td>
      <td id="T_a0baa_row65_col4" class="data row65 col4" ></td>
      <td id="T_a0baa_row65_col5" class="data row65 col5" ></td>
    </tr>
  </tbody>
</table>




```python

```
