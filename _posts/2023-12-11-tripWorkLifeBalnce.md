--- 
layout: single
title: "시각화 - 관광산업 워라밸과 매출액을 중심으로"
toc: true
toc_sticky: true
toc_label: "페이지 주요 목차"

---

### 관광산업 워라밸과 매출액을 분석
## 코로나 전후 워라밸 변화와 매출액 변화 시각



```python
import pandas as pd
import matplotlib.pyplot as plt
```

### 관광업 휴무일, 영업시간 분석


```python
year2020_1 = pd.read_excel('2020년관광산업특수분류.xlsx')
year2020_2 = pd.read_excel('2020년관광진흥법기준.xlsx')

year2021_1 = pd.read_excel('2021년관광산업관광진흥법기준 .xlsx')
year2021_2 = pd.read_excel('2021년관광산업특수분류기준.xlsx')
```


```python
year2020_11 = year2020_1[['Q8_2','Q8_3_1','Q8_3_2']] #이상치 없음
year2020_22 = year2020_2[['Q8_2','Q8_3_1','Q8_3_2']] # 이상치 5812,5813

year2021_11 = year2021_1[['Q8_2','Q8_3_1','Q8_3_2']]
year2021_22 = year2021_2[['Q8_2','Q8_3_1','Q8_3_2']]
```


```python
# 휴무일 여부  1 없음  , 2 있음  , 3 모름
year2020_11 = year2020_11.rename(columns={'Q8_2': '일일영업시간','Q8_3_1':'휴무일여부', 'Q8_3_2':'휴무일수'})
year2020_22 = year2020_22.rename(columns={'Q8_2': '일일영업시간','Q8_3_1':'휴무일여부', 'Q8_3_2':'휴무일수'})
year2021_11 = year2021_11.rename(columns={'Q8_2': '일일영업시간','Q8_3_1':'휴무일여부', 'Q8_3_2':'휴무일수'})
year2021_22 = year2021_22.rename(columns={'Q8_2': '일일영업시간','Q8_3_1':'휴무일여부', 'Q8_3_2':'휴무일수'})
```


```python
year2021_11[(year2021_11['휴무일여부'] == 2) & (year2021_11['휴무일수'] >= 1)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>일일영업시간</th>
      <th>휴무일여부</th>
      <th>휴무일수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>2</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7200</th>
      <td>8</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7206</th>
      <td>9</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>7207</th>
      <td>10</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7208</th>
      <td>10</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7210</th>
      <td>9</td>
      <td>2</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>3555 rows × 3 columns</p>
</div>




```python
year2020_11.to_excel('2020_특수.xlsx', index=False)
year2020_22.to_excel('2020_진흥법.xlsx', index=False)
year2021_11.to_excel('2021_진흥법.xlsx', index=False)
year2021_22.to_excel('2021_특수.xlsx', index=False)
```

#### 휴무일, 영업시간 분석


```python
import koreanize_matplotlib
year2020_11['일일영업시간'].plot.box()
```




    <Axes: >




    
![png](output_8_1.png)
    



```python
year2020_1 = pd.read_excel('2020년 기준 관광산업조사 DATA_PART2_관광산업특수분류 기준.xlsx')
year2021_1 = pd.read_excel('2021년 기준 관광산업조사 DATA PART2 관광산업특수분류 기준.xlsx')
```


```python
year2020_11 = year2020_1[['TYPE_2','H1_1','H1_2','H1_3','H1_4','H1_5','H1_6','H1_7','H1_8','Q8_2','Q8_3_1','Q8_3_2','Q10_3']] #이상치 없음
year2021_11 = year2021_1[['TYPE_2','H1_1','H1_2','H1_3','H1_4','H1_5','H1_6','H1_7','H1_8','Q8_2','Q8_3_1','Q8_3_2','Q10_3']] #이상치 없음
```


```python
year2020_11[year2020_11['Q8_2'] != 24]['TYPE_2'].value_counts()/len(year2020_11)   # 4번 관광숙박업, 3번 관광객이용시설업
```




    TYPE_2
    6    0.189499
    8    0.138209
    7    0.048643
    5    0.045445
    3    0.033973
    2    0.027686
    4    0.026803
    1    0.022281
    9    0.000772
    Name: count, dtype: float64




```python
year2021_11[year2021_11['Q8_2'] >= 9]['TYPE_2'].value_counts()/len(year2021_11)   # 4번 관광숙박업, 3번 관광객이용시설업
```




    TYPE_2
    4    0.327742
    8    0.106322
    3    0.091876
    6    0.035492
    5    0.030056
    2    0.027105
    1    0.013358
    7    0.010951
    9    0.001243
    Name: count, dtype: float64



#### 관광업 매출액 분석


```python
import numpy as np
```


```python
y2021 = year2021_11[year2021_11['Q8_2'] != 24].groupby('TYPE_2').agg({'Q8_2':np.mean, 'Q10_3':np.mean}).sort_values(by='Q8_2')
```

    C:\Users\User\AppData\Local\Temp\ipykernel_11812\1509577875.py:1: FutureWarning: The provided callable <function mean at 0x000001BD08A43820> is currently using SeriesGroupBy.mean. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "mean" instead.
      y2021 = year2021_11[year2021_11['Q8_2'] != 24].groupby('TYPE_2').agg({'Q8_2':np.mean, 'Q10_3':np.mean}).sort_values(by='Q8_2')
    


```python
y2020 = year2020_11[year2020_11['Q8_2'] != 24].groupby('TYPE_2').agg({'Q8_2':np.mean, 'Q10_3':np.mean}).sort_values(by='Q8_2')
```

    C:\Users\User\AppData\Local\Temp\ipykernel_11812\1151813236.py:1: FutureWarning: The provided callable <function mean at 0x000001BD08A43820> is currently using SeriesGroupBy.mean. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "mean" instead.
      y2020 = year2020_11[year2020_11['Q8_2'] != 24].groupby('TYPE_2').agg({'Q8_2':np.mean, 'Q10_3':np.mean}).sort_values(by='Q8_2')
    


```python
year2021_11[year2021_11['Q8_2'] != 24].groupby('TYPE_2').agg({'Q8_2':np.mean, 'Q10_3':np.mean}).sort_values(by='Q8_2')
```

    C:\Users\User\AppData\Local\Temp\ipykernel_11812\187672551.py:1: FutureWarning: The provided callable <function mean at 0x000001BD08A43820> is currently using SeriesGroupBy.mean. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "mean" instead.
      year2021_11[year2021_11['Q8_2'] != 24].groupby('TYPE_2').agg({'Q8_2':np.mean, 'Q10_3':np.mean}).sort_values(by='Q8_2')
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Q8_2</th>
      <th>Q10_3</th>
    </tr>
    <tr>
      <th>TYPE_2</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>7.636167</td>
      <td>60.849089</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.067974</td>
      <td>691.324183</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.603030</td>
      <td>30107.600000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8.816506</td>
      <td>1655.486831</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9.191489</td>
      <td>613.697872</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.283708</td>
      <td>647.078652</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.923977</td>
      <td>1197.742690</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11.088099</td>
      <td>10275.698609</td>
    </tr>
    <tr>
      <th>9</th>
      <td>16.375000</td>
      <td>117302.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
year2020_11[year2020_11['Q8_2'] != 24].groupby('TYPE_2').agg({'Q8_2':np.mean, 'Q10_3':np.mean}).sort_values(by='Q8_2')
```

    C:\Users\User\AppData\Local\Temp\ipykernel_11812\3374464940.py:1: FutureWarning: The provided callable <function mean at 0x000001BD08A43820> is currently using SeriesGroupBy.mean. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "mean" instead.
      year2020_11[year2020_11['Q8_2'] != 24].groupby('TYPE_2').agg({'Q8_2':np.mean, 'Q10_3':np.mean}).sort_values(by='Q8_2')
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Q8_2</th>
      <th>Q10_3</th>
    </tr>
    <tr>
      <th>TYPE_2</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>7.930442</td>
      <td>204.121653</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.197279</td>
      <td>633.317460</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.118812</td>
      <td>6890.549505</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9.120511</td>
      <td>2373.370311</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9.461165</td>
      <td>582.548544</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.889610</td>
      <td>648.938312</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.921811</td>
      <td>724.098765</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.087649</td>
      <td>23385.525896</td>
    </tr>
    <tr>
      <th>9</th>
      <td>17.857143</td>
      <td>72620.857143</td>
    </tr>
  </tbody>
</table>
</div>




```python
y2020_1 = y2020.loc[[7,1,5,9]]
y2020_2 = y2020.loc[[3]]

y2021_1 = y2021.loc[[7,1,5,9]]
y2021_2 = y2021.loc[[3]]
```


```python
y2020_1['년도'] = 2020
y2020_2['년도'] = 2020

y2021_1['년도'] = 2021
y2021_2['년도'] = 2021

y2020_1['산업분야'] = ['국제회의업','관광쇼핑업','관광편의시설업','카지노업']
y2020_2['산업분야'] = ['관광객이용시설업']

y2021_1['산업분야'] = ['국제회의업','관광쇼핑업','관광편의시설업','카지노업']
y2021_2['산업분야'] = ['관광객이용시설업']
```

### 시각화


```python
ss_1 = pd.concat([y2020_1, y2021_1]).reset_index(drop=True)
ss_2 = pd.concat([y2020_2, y2021_2]).reset_index(drop=True)
```


```python
ss_1 = ss_1.rename(columns={'Q8_2':'일일평균영업시간', 'Q10_3':'연간총매출액_전체(백만원)'})
ss_2 = ss_2.rename(columns={'Q8_2':'일일평균영업시간', 'Q10_3':'연간총매출액_전체(백만원)'})
```


```python
ss
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>일일평균영업시간</th>
      <th>연간총매출액_전체(백만원)</th>
      <th>년도</th>
      <th>산업분야</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.197279</td>
      <td>633.317460</td>
      <td>2020</td>
      <td>국제회의업</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.118812</td>
      <td>6890.549505</td>
      <td>2020</td>
      <td>관광쇼핑업</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.461165</td>
      <td>582.548544</td>
      <td>2020</td>
      <td>관광편의시설업</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.889610</td>
      <td>648.938312</td>
      <td>2020</td>
      <td>관광객이용시설업</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.857143</td>
      <td>72620.857143</td>
      <td>2020</td>
      <td>카지노업</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8.067974</td>
      <td>691.324183</td>
      <td>2021</td>
      <td>국제회의업</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8.603030</td>
      <td>30107.600000</td>
      <td>2021</td>
      <td>관광쇼핑업</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9.191489</td>
      <td>613.697872</td>
      <td>2021</td>
      <td>관광편의시설업</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9.923977</td>
      <td>1197.742690</td>
      <td>2021</td>
      <td>관광객이용시설업</td>
    </tr>
    <tr>
      <th>9</th>
      <td>16.375000</td>
      <td>117302.500000</td>
      <td>2021</td>
      <td>카지노업</td>
    </tr>
  </tbody>
</table>
</div>




```python
ss = pd.concat([y2020, y2021]).reset_index(drop=True)
ss = ss.rename(columns={'Q8_2':'일일평균영업시간', 'Q10_3':'연간총매출액_전체(백만원)'})
```


```python
ss.to_excel('2020~2021 산업분야별 연간총매출액.xlsx', index = False)
```


```python
y2018 = pd.read_excel('0.종합_18년기준 관광사업체조사.xlsx')
y2019 = pd.read_excel('2019년 기준 관광사업체조사_원자료.xlsx')
```


```python
y2018 = y2018[['SQ3', 'Q8', 'Q17_W_B']]
y2018 = y2018.rename(columns={'SQ3':'산업분야', 'Q8':'일일평균영업시간', 'Q17_W_B':'연간총매출액_전체(백만원)'})
y2019 = y2019[['SQ3', 'Q8', 'Q17_W_B']]
y2019 = y2019.rename(columns={'SQ3':'산업분야', 'Q8':'일일평균영업시간', 'Q17_W_B':'연간총매출액_전체(백만원)'})
```


```python
y2018 = y2018[y2018['일일평균영업시간'] != 24].groupby('산업분야').agg({'일일평균영업시간':np.mean, '연간총매출액_전체(백만원)':np.mean}).sort_values(by='일일평균영업시간')
y2019 = y2019[y2019['일일평균영업시간'] != 24].groupby('산업분야').agg({'일일평균영업시간':np.mean, '연간총매출액_전체(백만원)':np.mean}).sort_values(by='일일평균영업시간')
```

    C:\Users\User\AppData\Local\Temp\ipykernel_11812\1880333426.py:1: FutureWarning: The provided callable <function mean at 0x000001BD08A43820> is currently using SeriesGroupBy.mean. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "mean" instead.
      y2018 = y2018[y2018['일일평균영업시간'] != 24].groupby('산업분야').agg({'일일평균영업시간':np.mean, '연간총매출액_전체(백만원)':np.mean}).sort_values(by='일일평균영업시간')
    C:\Users\User\AppData\Local\Temp\ipykernel_11812\1880333426.py:2: FutureWarning: The provided callable <function mean at 0x000001BD08A43820> is currently using SeriesGroupBy.mean. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "mean" instead.
      y2019 = y2019[y2019['일일평균영업시간'] != 24].groupby('산업분야').agg({'일일평균영업시간':np.mean, '연간총매출액_전체(백만원)':np.mean}).sort_values(by='일일평균영업시간')
    


```python
y2018 = y2018.loc[[7,2]]
y2019 = y2019.loc[[7,2]]

y2018['년도'] = 2018
y2019['년도'] = 2019

y2018['산업분야'] = ['관광편의시설업','관광숙박업']
y2019['산업분야'] = ['관광편의시설업','관광숙박업']
```


```python
sss = pd.concat([y2018, y2019]).reset_index(drop=True)
```


```python
sss.to_excel('2018~2019 산업분야별 연간총매출액.xlsx', index = False)
```


```python
# 영업시간은 증가, 매출액은 감소     4,1,6   4국체회의업, 1여행업, 6 유원시설업
# 영업시간은 감소, 매출액은 증가     7,2     7관광편의시설업,  2관광숙박업
```


```python
plt.figure(figsize=(10, 6))
sns.barplot(x='산업분야', y='일일평균영업시간', hue='년도', data=sss, dodge=True, palette=['lightgray','skyblue'], alpha=0.8)
```

    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    




    <Axes: xlabel='산업분야', ylabel='일일평균영업시간'>




    
![png](output_34_2.png)
    



```python
plt.figure(figsize=(10, 6))
sns.barplot(x='산업분야', y='연간총매출액_전체(백만원)', hue='년도', data=sss, dodge=True, palette=['lightgray','lightcoral'], alpha=0.8)
```

    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    




    <Axes: xlabel='산업분야', ylabel='연간총매출액_전체(백만원)'>




    
![png](output_35_2.png)
    



```python
import koreanize_matplotlib
plt.figure(figsize=(10, 6))
sns.barplot(x='산업분야', y='일일평균영업시간', hue='년도', data=ss_1, dodge=True, palette=['lightgray','skyblue'], alpha=0.8)
```

    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    




    <Axes: xlabel='산업분야', ylabel='일일평균영업시간'>




    
![png](output_36_2.png)
    



```python
import koreanize_matplotlib

plt.figure(figsize=(10, 6))
sns.barplot(x='산업분야', y='연간총매출액_전체(백만원)', hue='년도', data=ss_1, dodge=True, palette=['lightgray','lightcoral'], alpha=0.8)
plt.ylim(0,80000)
```

    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    




    (0.0, 80000.0)




    
![png](output_37_2.png)
    



```python
plt.figure(figsize=(10, 6))
sns.barplot(x='산업분야', y='일일평균영업시간', hue='년도', data=ss_2, dodge=True, palette=['lightgray','skyblue'], alpha=0.8)
plt.ylim(9.5,10)
```

    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    




    (9.5, 10.0)




    
![png](output_38_2.png)
    



```python
plt.figure(figsize=(10, 6))
sns.barplot(x='산업분야', y='연간총매출액_전체(백만원)', hue='년도', data=ss_2, dodge=True, palette=['lightgray','lightcoral'], alpha=0.8)
```

    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\User\anaconda3\envs\MY_PYTHON\lib\site-packages\seaborn\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    




    <Axes: xlabel='산업분야', ylabel='연간총매출액_전체(백만원)'>




    
![png](output_39_2.png)
    

### 관광업 종사자의 워라벨 분석


```python
import pandas as pd
year2020 = pd.read_excel('2020.xlsx')
year2021 = pd.read_excel('2021.xlsx')
```


```python
year2020
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>일일영업시간</th>
      <th>휴무일여부</th>
      <th>휴무일수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.0</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.0</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.0</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.0</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9061</th>
      <td>16.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9062</th>
      <td>24.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9063</th>
      <td>24.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9064</th>
      <td>17.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9065</th>
      <td>18.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>9066 rows × 3 columns</p>
</div>




```python
year2021
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>일일영업시간</th>
      <th>휴무일여부</th>
      <th>휴무일수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>2</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12871</th>
      <td>7</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12872</th>
      <td>7</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12873</th>
      <td>6</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12874</th>
      <td>7</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12875</th>
      <td>6</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>12876 rows × 3 columns</p>
</div>




```python
import seaborn as sns
import matplotlib.pyplot as plt
import koreanize_matplotlib
```

### 시각화


```python
year2020_worktime = year2020['일일영업시간'][year2020['일일영업시간'] != 24].mean()
year2021_worktime = year2021['일일영업시간'][year2021['일일영업시간'] != 24].mean()

year2020_holyday = year2020['휴무일수'].mean()
year2021_holyday = year2021['휴무일수'].mean()
```


```python
data = pd.DataFrame({'2020' : [year2020_worktime, year2020_holyday],
             '2021' : [year2021_worktime, year2021_holyday]},
            index = ['일일영업시간','휴무일수'])
```


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2020</th>
      <th>2021</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>일일영업시간</th>
      <td>9.004862</td>
      <td>8.668089</td>
    </tr>
    <tr>
      <th>휴무일수</th>
      <td>2.678469</td>
      <td>2.820596</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the bar graph
plt.bar(data.columns, data.iloc[1], color=['lightgray','lightcoral'], label = [2020,2021])

# Customize the plot
plt.xlabel('년도')
plt.ylabel('일수')
plt.title('평균 휴무일수')
plt.ylim(2.0,3.0)  # Rotate x-axis labels if needed
plt.legend()
# plt.grid()
plt.show()
```


    
![png](output_9_0.png)
    



```python
# Plot the bar graph
plt.bar(data.columns, data.iloc[0], color=['lightgray','skyblue'], label = [2020,2021])

# Customize the plot
plt.xlabel('년도')
plt.ylabel('시간')
plt.title('평균 일일영업시간')
plt.ylim(8.2,9.2)  # Rotate x-axis labels if needed
plt.legend()
# plt.grid()
plt.show()
```


    
![png](output_10_0.png)
    



```python
year2018 = pd.read_excel('2018.xlsx')
year2019 = pd.read_excel('2019.xlsx')
```


```python
year2018_worktime = year2018['일일영업시간'][year2018['일일영업시간'] != 24].mean()
year2019_worktime = year2019['일일영업시간'][year2019['일일영업시간'] != 24].mean()

year2018_holyday = year2018['휴무일수'].mean()
year2019_holyday = year2019['휴무일수'].mean()
```


```python
data = pd.DataFrame({'2018' : [year2018_worktime, year2018_holyday],
             '2019' : [year2019_worktime, year2019_holyday]},
            index = ['일일영업시간','휴무일수'])
```


```python
# Plot the bar graph
plt.bar(data.columns, data.iloc[1], color=['lightgray','lightcoral'], label = [2018,2019])

# Customize the plot
plt.xlabel('년도')
plt.ylabel('일수')
plt.title('평균 휴무일수')
plt.ylim(2.2,3.2)  # Rotate x-axis labels if needed
plt.legend()
# plt.grid()
plt.show()
```


    
![png](output_14_0.png)
    



```python
# Plot the bar graph
plt.bar(data.columns, data.iloc[0], color=['lightgray','skyblue'], label = [2018,2019])

# Customize the plot
plt.xlabel('년도')
plt.ylabel('시간')
plt.title('평균 일일영업시간')
plt.ylim(8.5,9.5)  # Rotate x-axis labels if needed
plt.legend()
# plt.grid()
plt.show()
```


    
![png](output_15_0.png)
    



```python
data = pd.DataFrame({'2018' : [year2018_worktime, year2018_holyday]
            ,'2019' : [year2019_worktime, year2019_holyday]
            ,'2020' : [year2020_worktime, year2020_holyday],
             '2021' : [year2021_worktime, year2021_holyday]},
            index = ['일일영업시간','휴무일수'])
```


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2018</th>
      <th>2019</th>
      <th>2020</th>
      <th>2021</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>일일영업시간</th>
      <td>9.447409</td>
      <td>9.137001</td>
      <td>9.004862</td>
      <td>8.668089</td>
    </tr>
    <tr>
      <th>휴무일수</th>
      <td>2.463592</td>
      <td>2.991686</td>
      <td>2.678469</td>
      <td>2.820596</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the bar graph
plt.bar(data.columns, data.iloc[1], color=['lightgray','lightgray','lightgray','lightcoral'])

plt.plot(data.columns, data.iloc[1], marker='o', linestyle='-', color='black')

# Customize the plot
plt.xlabel('년도')
plt.ylabel('일수')
plt.title('평균 휴무일수')
plt.ylim(2, 3.5)  # Rotate x-axis labels if needed
#plt.legend()
# plt.grid()
plt.show()
```


    
![png](output_18_0.png)
    



```python
# Plot the bar graph
plt.bar(data.columns, data.iloc[0], color=['lightgray','lightgray','lightgray','skyblue'])

plt.plot(data.columns, data.iloc[0], marker='o', linestyle='-', color='black')

# Customize the plot
plt.xlabel('년도')
plt.ylabel('일수')
plt.title('평균 일일영업일수')
plt.ylim(8, 10)  # Rotate x-axis labels if needed
#plt.legend()
# plt.grid()
plt.show()
```


    
![png](output_19_0.png)
    

