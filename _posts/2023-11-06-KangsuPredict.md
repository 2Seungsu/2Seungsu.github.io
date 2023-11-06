--- 
layout: single
title: "강수량 분석"
---


```python
import pandas as pd
import matplotlib.pyplot as plt
```
## 수온, 기온, 강수량의 상관관계와 시각화 및 회귀분석 
### 인천


```python
kangsu_incheon = pd.read_csv('인천.csv',skiprows=7,encoding='cp949')
kangsu_kunsan = pd.read_csv('군산.csv',skiprows=7,encoding='cp949')
```


```python
## 년도별 평균 데이터
kangsu_incheon['year'] = kangsu_incheon['년월'].apply(lambda x: x.split('-')[0])
kangsu_incheon['year']=kangsu_incheon['year'].astype('int')  
```


```python
kangsu_incheon = kangsu_incheon[kangsu_incheon['year'] >= 2019]
kangsu_incheon.reset_index(inplace = True, drop = True)
```


```python
incheon=pd.read_csv('incheon.csv')
kunsan=pd.read_csv('kunsan.csv')
```


```python
incheon['강수량'] = kangsu_incheon['강수량(mm)']
incheon['수온'] = incheon['mean']
```


```python
incheon_kion = pd.read_csv('인천_기온.csv',skiprows=7,encoding='cp949')
#2023년 8월 결측치 대체
incheon_kion[incheon_kion['년월'].str.contains('08')]['평균최고기온(℃)'].mean()
```




    28.575




```python
incheon['기온']=incheon_kion['평균최고기온(℃)'].fillna(incheon_kion[incheon_kion['년월'].str.contains('08')]['평균최고기온(℃)'].mean())
incheon[['수온','기온','강수량']].corr()
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
      <th>수온</th>
      <th>기온</th>
      <th>강수량</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>수온</th>
      <td>1.000000</td>
      <td>0.944807</td>
      <td>0.647211</td>
    </tr>
    <tr>
      <th>기온</th>
      <td>0.944807</td>
      <td>1.000000</td>
      <td>0.621228</td>
    </tr>
    <tr>
      <th>강수량</th>
      <td>0.647211</td>
      <td>0.621228</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### 군산


```python
## 년도별 평균 데이터
kangsu_kunsan['year'] = kangsu_kunsan['년월'].apply(lambda x: x.split('-')[0])
kangsu_kunsan['year'] = kangsu_kunsan['year'].astype('int')  
```


```python
kangsu_kunsan = kangsu_kunsan[kangsu_kunsan['year'] >= 2019]
kangsu_kunsan.reset_index(inplace = True, drop = True)
```


```python
kunsan['강수량'] = kangsu_kunsan['강수량(mm)']
kunsan_kion = pd.read_csv('군산_기온.csv',skiprows=7,encoding='cp949')
#2023년 8월 결측치 대체
kunsan['기온']=kunsan_kion['평균최고기온(℃)'].fillna(kunsan_kion[kunsan_kion['년월'].str.contains('08')]['평균최고기온(℃)'].mean())
kunsan['수온'] = kunsan['mean']
```


```python
kunsan[['수온','기온','강수량']].corr()
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
      <th>수온</th>
      <th>기온</th>
      <th>강수량</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>수온</th>
      <td>1.000000</td>
      <td>0.957953</td>
      <td>0.550803</td>
    </tr>
    <tr>
      <th>기온</th>
      <td>0.957953</td>
      <td>1.000000</td>
      <td>0.572261</td>
    </tr>
    <tr>
      <th>강수량</th>
      <td>0.550803</td>
      <td>0.572261</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



기온, 수온, 강수량 모두 양의 상관관계가 있다.

#### 회귀분석


```python
import statsmodels.api as sm
```


```python
X = sm.add_constant(incheon[['강수량','기온']])
model = sm.OLS(incheon['mean'], X)
model.fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>mean</td>       <th>  R-squared:         </th> <td>   0.899</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.895</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   234.8</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 20 Aug 2023</td> <th>  Prob (F-statistic):</th> <td>4.60e-27</td>
</tr>
<tr>
  <th>Time:</th>                 <td>09:55:37</td>     <th>  Log-Likelihood:    </th> <td> -133.47</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    56</td>      <th>  AIC:               </th> <td>   272.9</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    53</td>      <th>  BIC:               </th> <td>   279.0</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>    0.5766</td> <td>    0.758</td> <td>    0.761</td> <td> 0.450</td> <td>   -0.944</td> <td>    2.097</td>
</tr>
<tr>
  <th>강수량</th>   <td>    0.0066</td> <td>    0.004</td> <td>    1.758</td> <td> 0.084</td> <td>   -0.001</td> <td>    0.014</td>
</tr>
<tr>
  <th>기온</th>    <td>    0.7843</td> <td>    0.050</td> <td>   15.833</td> <td> 0.000</td> <td>    0.685</td> <td>    0.884</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 8.537</td> <th>  Durbin-Watson:     </th> <td>   0.516</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.014</td> <th>  Jarque-Bera (JB):  </th> <td>   2.885</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.153</td> <th>  Prob(JB):          </th> <td>   0.236</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 1.931</td> <th>  Cond. No.          </th> <td>    339.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
model1 = sm.OLS(incheon['강수량'],sm.add_constant(incheon['수온']))
model1.fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>강수량</td>       <th>  R-squared:         </th> <td>   0.419</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.408</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   38.92</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 20 Aug 2023</td> <th>  Prob (F-statistic):</th> <td>7.04e-08</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:14:33</td>     <th>  Log-Likelihood:    </th> <td> -333.41</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    56</td>      <th>  AIC:               </th> <td>   670.8</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    54</td>      <th>  BIC:               </th> <td>   674.9</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>  -35.2358</td> <td>   25.657</td> <td>   -1.373</td> <td> 0.175</td> <td>  -86.676</td> <td>   16.204</td>
</tr>
<tr>
  <th>수온</th>    <td>    9.6054</td> <td>    1.540</td> <td>    6.239</td> <td> 0.000</td> <td>    6.519</td> <td>   12.692</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>39.284</td> <th>  Durbin-Watson:     </th> <td>   2.004</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 131.109</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.942</td> <th>  Prob(JB):          </th> <td>3.39e-29</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 9.411</td> <th>  Cond. No.          </th> <td>    33.8</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
model2 = sm.OLS(incheon['강수량'],sm.add_constant(incheon['기온']))
model2.fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>강수량</td>       <th>  R-squared:         </th> <td>   0.386</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.375</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   33.94</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 20 Aug 2023</td> <th>  Prob (F-statistic):</th> <td>3.24e-07</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:14:30</td>     <th>  Log-Likelihood:    </th> <td> -334.95</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    56</td>      <th>  AIC:               </th> <td>   673.9</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    54</td>      <th>  BIC:               </th> <td>   678.0</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>  -34.0209</td> <td>   27.030</td> <td>   -1.259</td> <td> 0.214</td> <td>  -88.212</td> <td>   20.170</td>
</tr>
<tr>
  <th>기온</th>    <td>    8.1819</td> <td>    1.404</td> <td>    5.826</td> <td> 0.000</td> <td>    5.366</td> <td>   10.998</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>44.063</td> <th>  Durbin-Watson:     </th> <td>   1.962</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 176.314</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 2.135</td> <th>  Prob(JB):          </th> <td>5.18e-39</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>10.572</td> <th>  Cond. No.          </th> <td>    40.0</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
model2 = sm.OLS(kunsan['강수량'],sm.add_constant(kunsan['기온']))
model2.fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>강수량</td>       <th>  R-squared:         </th> <td>   0.327</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.315</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   26.30</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 20 Aug 2023</td> <th>  Prob (F-statistic):</th> <td>4.07e-06</td>
</tr>
<tr>
  <th>Time:</th>                 <td>17:24:53</td>     <th>  Log-Likelihood:    </th> <td> -347.90</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    56</td>      <th>  AIC:               </th> <td>   699.8</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    54</td>      <th>  BIC:               </th> <td>   703.9</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>  -65.0082</td> <td>   38.844</td> <td>   -1.674</td> <td> 0.100</td> <td> -142.885</td> <td>   12.869</td>
</tr>
<tr>
  <th>기온</th>    <td>    9.7206</td> <td>    1.896</td> <td>    5.128</td> <td> 0.000</td> <td>    5.920</td> <td>   13.521</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>63.413</td> <th>  Durbin-Watson:     </th> <td>   1.936</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 484.941</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 3.042</td> <th>  Prob(JB):          </th> <td>4.97e-106</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>16.070</td> <th>  Cond. No.          </th> <td>    48.5</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
model2 = sm.OLS(kunsan['강수량'],sm.add_constant(kunsan['수온']))
model2.fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>강수량</td>       <th>  R-squared:         </th> <td>   0.303</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.290</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   23.52</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 20 Aug 2023</td> <th>  Prob (F-statistic):</th> <td>1.09e-05</td>
</tr>
<tr>
  <th>Time:</th>                 <td>17:25:13</td>     <th>  Log-Likelihood:    </th> <td> -348.89</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    56</td>      <th>  AIC:               </th> <td>   701.8</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    54</td>      <th>  BIC:               </th> <td>   705.8</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>  -44.1989</td> <td>   36.930</td> <td>   -1.197</td> <td> 0.237</td> <td> -118.240</td> <td>   29.842</td>
</tr>
<tr>
  <th>수온</th>    <td>   10.6187</td> <td>    2.190</td> <td>    4.850</td> <td> 0.000</td> <td>    6.229</td> <td>   15.009</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>64.954</td> <th>  Durbin-Watson:     </th> <td>   1.887</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 515.332</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 3.131</td> <th>  Prob(JB):          </th> <td>1.25e-112</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>16.477</td> <th>  Cond. No.          </th> <td>    37.4</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



유의확률이 유의수준 0.05보다 작으므로 귀무가설을 기각할 수 있다.


```python
from plotnine import *
```


```python
from matplotlib import font_manager, rc
font_path = r'C:\Users\LG\Desktop\exam_pandas\Day_07'+'\malgun.ttf'
font_name= font_manager.FontProperties(fname=font_path).get_name()
rc('font', family = font_name)
```


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
# 인천
ggplot(incheon) + geom_point(aes(x='기온',y='강수량',colour='mean'),alpha=0.5) + geom_smooth(aes(x='mean',y='강수량', colour='mean'),method='lm') +\
scale_x_continuous(breaks=range(-10, 40, 5))  + ylim(0,350) + theme(text=element_text(fontproperties=font_name))
```


    
![output_26_0](https://github.com/2Seungsu/AI-BigData_curriculum/assets/141051562/e40978da-b0d4-49f4-9686-5c169d039921)

    





    <Figure Size: (640 x 480)>




```python
# 인천
ggplot(incheon) + geom_point(aes(x='기온',y='강수량',colour='mean'),alpha=0.5) + geom_smooth(aes(x='기온',y='강수량'),method='lm') +\
scale_x_continuous(breaks=range(-10, 40, 5))  + ylim(0,350)  + theme(text=element_text(fontproperties=font_name))
```


    
![output_27_0](https://github.com/2Seungsu/AI-BigData_curriculum/assets/141051562/16b8c785-2c2e-4cc4-a518-cb717378023b)
    





    <Figure Size: (640 x 480)>




```python
# 군산
ggplot(kunsan) + geom_point(aes(x='기온',y='강수량',colour='mean'),alpha=0.5) + geom_smooth(aes(x='mean',y='강수량', colour='mean'),method='lm')+\
scale_x_continuous(breaks=range(-10, 40, 5))  + ylim(0,350) + theme(text=element_text(fontproperties=font_name))
```


    
![output_28_0](https://github.com/2Seungsu/AI-BigData_curriculum/assets/141051562/a7c61e42-ddde-41fa-95d6-904d30b6f861)
    





    <Figure Size: (640 x 480)>




```python
# 군산
ggplot(kunsan) + geom_point(aes(x='기온',y='강수량',colour='mean'),alpha=0.5) + geom_smooth(aes(x='기온',y='강수량'),method='lm')+\
scale_x_continuous(breaks=range(-10, 40, 5))  + ylim(0,350) + theme(text=element_text(fontproperties=font_name))
```


    
![output_29_0](https://github.com/2Seungsu/AI-BigData_curriculum/assets/141051562/37c2c88e-fa6f-4b20-9ad4-82a3c16db789)
    





    <Figure Size: (640 x 480)>



산점도와 회귀분석 결과 기온과 수온 인관관계가 있다고 볼 수 있다. 그리고 수온 상승으로 인한 강수량 증가를 대비할 필요가 있다.
