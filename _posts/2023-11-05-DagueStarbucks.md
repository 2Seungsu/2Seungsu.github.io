--- 
layout: single
title: "대구 스타벅스 입지 분석"
toc: true
toc: true
toc_sticky: true
toc_label: "페이지 주요 목차"
---

## 대구_달성군,남구,서구,동구 스타벅스 분석


```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
```

### 스타벅스 주변 유동인구 분석


```python
df = pd.read_csv('analyze_fourGu_21.csv')
df.sort_values(by='하루평균유동인구_대중교통',inplace=True)
```


```python
df
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
      <th>매장명</th>
      <th>위도</th>
      <th>경도</th>
      <th>인접_상가수</th>
      <th>인접_학교수</th>
      <th>인접_지하철역수</th>
      <th>하루평균유동인구_지하철</th>
      <th>인접_버스정류장수</th>
      <th>하루평균유동인구_버스</th>
      <th>하루평균유동인구_대중교통</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>대구팔공산</td>
      <td>35.987868</td>
      <td>128.636347</td>
      <td>19</td>
      <td>1</td>
      <td>0</td>
      <td>0.00</td>
      <td>4</td>
      <td>31.81</td>
      <td>31.81</td>
    </tr>
    <tr>
      <th>10</th>
      <td>대구동촌유원지</td>
      <td>35.880110</td>
      <td>128.653159</td>
      <td>104</td>
      <td>8</td>
      <td>0</td>
      <td>0.00</td>
      <td>3</td>
      <td>62.59</td>
      <td>62.59</td>
    </tr>
    <tr>
      <th>1</th>
      <td>대구앞산DT</td>
      <td>35.835516</td>
      <td>128.579871</td>
      <td>152</td>
      <td>4</td>
      <td>0</td>
      <td>0.00</td>
      <td>6</td>
      <td>73.84</td>
      <td>73.84</td>
    </tr>
    <tr>
      <th>8</th>
      <td>대구봉무</td>
      <td>35.921942</td>
      <td>128.639708</td>
      <td>513</td>
      <td>5</td>
      <td>0</td>
      <td>0.00</td>
      <td>2</td>
      <td>103.48</td>
      <td>103.48</td>
    </tr>
    <tr>
      <th>12</th>
      <td>대구율하</td>
      <td>35.863537</td>
      <td>128.693927</td>
      <td>219</td>
      <td>6</td>
      <td>0</td>
      <td>0.00</td>
      <td>5</td>
      <td>110.41</td>
      <td>110.41</td>
    </tr>
    <tr>
      <th>6</th>
      <td>동대구로DT</td>
      <td>35.868087</td>
      <td>128.627159</td>
      <td>299</td>
      <td>7</td>
      <td>0</td>
      <td>0.00</td>
      <td>3</td>
      <td>163.26</td>
      <td>163.26</td>
    </tr>
    <tr>
      <th>14</th>
      <td>대구테크노폴리스</td>
      <td>35.693334</td>
      <td>128.459251</td>
      <td>226</td>
      <td>6</td>
      <td>0</td>
      <td>0.00</td>
      <td>4</td>
      <td>169.44</td>
      <td>169.44</td>
    </tr>
    <tr>
      <th>2</th>
      <td>대구가톨릭대학교병원</td>
      <td>35.843390</td>
      <td>128.567422</td>
      <td>161</td>
      <td>4</td>
      <td>0</td>
      <td>0.00</td>
      <td>2</td>
      <td>207.10</td>
      <td>207.10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>대구평리DT</td>
      <td>35.866138</td>
      <td>128.555343</td>
      <td>297</td>
      <td>13</td>
      <td>0</td>
      <td>0.00</td>
      <td>4</td>
      <td>548.65</td>
      <td>548.65</td>
    </tr>
    <tr>
      <th>11</th>
      <td>반야월이마트</td>
      <td>35.870899</td>
      <td>128.727397</td>
      <td>172</td>
      <td>7</td>
      <td>1</td>
      <td>7231.25</td>
      <td>4</td>
      <td>74.93</td>
      <td>7306.18</td>
    </tr>
    <tr>
      <th>13</th>
      <td>대구각산역DT</td>
      <td>35.868908</td>
      <td>128.726121</td>
      <td>235</td>
      <td>6</td>
      <td>1</td>
      <td>7231.25</td>
      <td>8</td>
      <td>138.09</td>
      <td>7369.34</td>
    </tr>
    <tr>
      <th>0</th>
      <td>대구영대병원역DT</td>
      <td>35.845522</td>
      <td>128.593536</td>
      <td>237</td>
      <td>3</td>
      <td>1</td>
      <td>9028.99</td>
      <td>2</td>
      <td>273.84</td>
      <td>9302.83</td>
    </tr>
    <tr>
      <th>15</th>
      <td>대구죽곡</td>
      <td>35.859055</td>
      <td>128.464166</td>
      <td>448</td>
      <td>9</td>
      <td>1</td>
      <td>11544.58</td>
      <td>5</td>
      <td>425.33</td>
      <td>11969.91</td>
    </tr>
    <tr>
      <th>9</th>
      <td>동대구터미널</td>
      <td>35.876315</td>
      <td>128.628590</td>
      <td>394</td>
      <td>4</td>
      <td>1</td>
      <td>31005.62</td>
      <td>4</td>
      <td>1465.97</td>
      <td>32471.59</td>
    </tr>
    <tr>
      <th>4</th>
      <td>신세계대구3F(티바나)</td>
      <td>35.877767</td>
      <td>128.628498</td>
      <td>361</td>
      <td>4</td>
      <td>1</td>
      <td>31005.62</td>
      <td>4</td>
      <td>1826.74</td>
      <td>32832.36</td>
    </tr>
    <tr>
      <th>5</th>
      <td>신세계대구8FR</td>
      <td>35.877812</td>
      <td>128.628453</td>
      <td>359</td>
      <td>4</td>
      <td>1</td>
      <td>31005.62</td>
      <td>4</td>
      <td>1826.74</td>
      <td>32832.36</td>
    </tr>
  </tbody>
</table>
</div>



### 시각화 및 분석


```python
from plotnine import *
```


```python
#한글 폰트
from matplotlib import font_manager, rc
font_path = r'C:\Users\LG\Desktop\exam_pandas\Day_07'+'\malgun.ttf'
font_name= font_manager.FontProperties(fname=font_path).get_name()
rc('font', family = font_name)
```


```python
p1 = ggplot(df) + geom_bar(aes(x='매장명',y='하루평균유동인구_대중교통',fill = '인접_상가수'),stat='identity') + theme(text=element_text(fontproperties=font_name),axis_text_x=element_text(angle=60, hjust=1)) + geom_hline(yintercept=782.54, linetype='dashed', color='red')
```


```python
print(p1)
```


    
![output_9_0](https://github.com/2Seungsu/AI-BigData_curriculum/assets/141051562/c60c8a69-51c9-4e32-b5e5-55365fda92e1)
    


    
    

인접 상가 수와 하루평균 유동인구가 많은 곳에도 스타벅스가 들어섰지만 유동인구 적은데 스타벅스가 입지한 곳이 9지점이 있다.(빨간선은 대구 전체 스타벅스의 중앙값)

다른 요인이 뭐가 있을가 생각하니 유동인구가 있어서 유동인구를 그래프에 추가하여 시각화를 하였다.



```python
ggplot(df) + geom_bar(aes(x='매장명',y='인접_상가수',fill='하루평균유동인구_대중교통'),stat='identity') + theme(text=element_text(fontproperties=font_name),axis_text_x=element_text(angle=70, hjust=1))+ geom_hline(yintercept=347, linetype='dashed', color='red')
```


    
![output_11_0](https://github.com/2Seungsu/AI-BigData_curriculum/assets/141051562/0bd2254c-db66-4e51-a0be-f4fb5d0949c0)
    





    <Figure Size: (640 x 480)>



스타벅스 지점은 인접상가 수가 약 100개 이상인 곳이 대부분이다.


```python
ggplot(df) + geom_bar(aes(x='매장명',y='인접_학교수',fill='하루평균유동인구_대중교통'),stat='identity') + theme(text=element_text(fontproperties=font_name),axis_text_x=element_text(angle=70, hjust=1)) + geom_hline(yintercept=7.7, linetype='dashed', color='red')
```


    
![output_13_0](https://github.com/2Seungsu/AI-BigData_curriculum/assets/141051562/3bd43f6d-410c-4a25-bb83-9c1b3aa51117)
    





    <Figure Size: (640 x 480)>



스타벅스는 주변에 상가와 학교가 많은 곳에 들어서는 것 같다.

스타벅스 지점 인접 학교 수를 비교해보니 상가수가 적은 지점이랑 겹친 지점이 있는 것 같다.


```python
fig, axs = plt.subplots(1,2,figsize= (15,6))
axs[0].bar(df['매장명'],df['인접_상가수'])
axs[1].bar(df['매장명'],df['인접_학교수'])
axs[0].set_xticks(range(len(df['매장명'])))
axs[1].set_xticks(range(len(df['매장명'])))
axs[0].set_xticklabels(df['매장명'].to_list(), rotation=42,ha='right')
axs[1].set_xticklabels(df['매장명'].to_list(), rotation=42,ha='right')
axs[0].set_title('인접_상가수')
axs[1].set_title('인접_학교수')
axs[0].axhline(347,color='red')
axs[1].axhline(7.7,color='red')
fig.tight_layout()
plt.show()
```


    
![output_16_0](https://github.com/2Seungsu/AI-BigData_curriculum/assets/141051562/c2704d42-bd59-44d7-9a7e-2bff0218dc3f)
    


인접 학교 수와 인접 매장이 적은 지점 중 겹치는 매장이 있는지 확인하고, 그 매장을 학교, 매장 수와는 다른 입지 조건이라고 판단한다.
인접 학교 수와 인접 매장수가 적은 스타벅스 매장이 가지고 있는 특징을 찾아본다.


```python
mask = (df['인접_상가수'] <= 347) & (df['인접_학교수'] <= 7.7 ) & (df['하루평균유동인구_대중교통'] < 782.54)
```


```python
smallDF=df[mask]
smallDF
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
      <th>매장명</th>
      <th>위도</th>
      <th>경도</th>
      <th>인접_상가수</th>
      <th>인접_학교수</th>
      <th>인접_지하철역수</th>
      <th>하루평균유동인구_지하철</th>
      <th>인접_버스정류장수</th>
      <th>하루평균유동인구_버스</th>
      <th>하루평균유동인구_대중교통</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>대구팔공산</td>
      <td>35.987868</td>
      <td>128.636347</td>
      <td>19</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>31.81</td>
      <td>31.81</td>
    </tr>
    <tr>
      <th>1</th>
      <td>대구앞산DT</td>
      <td>35.835516</td>
      <td>128.579871</td>
      <td>152</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>73.84</td>
      <td>73.84</td>
    </tr>
    <tr>
      <th>12</th>
      <td>대구율하</td>
      <td>35.863537</td>
      <td>128.693927</td>
      <td>219</td>
      <td>6</td>
      <td>0</td>
      <td>0.0</td>
      <td>5</td>
      <td>110.41</td>
      <td>110.41</td>
    </tr>
    <tr>
      <th>6</th>
      <td>동대구로DT</td>
      <td>35.868087</td>
      <td>128.627159</td>
      <td>299</td>
      <td>7</td>
      <td>0</td>
      <td>0.0</td>
      <td>3</td>
      <td>163.26</td>
      <td>163.26</td>
    </tr>
    <tr>
      <th>14</th>
      <td>대구테크노폴리스</td>
      <td>35.693334</td>
      <td>128.459251</td>
      <td>226</td>
      <td>6</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>169.44</td>
      <td>169.44</td>
    </tr>
    <tr>
      <th>2</th>
      <td>대구가톨릭대학교병원</td>
      <td>35.843390</td>
      <td>128.567422</td>
      <td>161</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>207.10</td>
      <td>207.10</td>
    </tr>
  </tbody>
</table>
</div>



대구팔공산, 대구앞산DT, 대구카톡릭대학교병원, 대구율하, 동대로DT, 대구테크노폴리스 스타벅스 지점의 입지조건을 따로 알아본다.

- 대구 팔공산 : 관광객이나 자연과 아름다운 조화를 목적으로 입지한 것으로 보인다. 대구관광코스로도 공공데이터포털에 올라와있다.


```python
#공공데이터 대구광역시_관광코스
palgong = pd.read_csv('대구광역시_관광코스 정보_20210906.csv',encoding='cp949')
```


```python
palgong[palgong['코스 주제'] == '팔공산'][['코스타이틀','지역','관광지']]
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
      <th>코스타이틀</th>
      <th>지역</th>
      <th>관광지</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29</th>
      <td>팔공산 힐링</td>
      <td>동구</td>
      <td>갓바위</td>
    </tr>
    <tr>
      <th>30</th>
      <td>팔공산 힐링</td>
      <td>동구</td>
      <td>아양기찻길</td>
    </tr>
    <tr>
      <th>31</th>
      <td>팔공산 힐링</td>
      <td>동구</td>
      <td>동화사</td>
    </tr>
    <tr>
      <th>32</th>
      <td>팔공산 힐링</td>
      <td>동구</td>
      <td>방짜유기박물관</td>
    </tr>
    <tr>
      <th>33</th>
      <td>팔공산 힐링</td>
      <td>동구</td>
      <td>시민안전테마파크</td>
    </tr>
    <tr>
      <th>34</th>
      <td>팔공산 힐링</td>
      <td>동구</td>
      <td>동화사 템플스테이</td>
    </tr>
    <tr>
      <th>35</th>
      <td>팔공산 힐링</td>
      <td>동구</td>
      <td>팔공산 단풍축제</td>
    </tr>
  </tbody>
</table>
</div>



- 대구앞산DT: 지점 위치가 앞산 카페거리에 있다. 카페거리에 카페와 음식점 등이 몰려있기 때문에 스타벅스가 들어온것으로 보인다.


```python
apsan = pd.read_csv('download.csv')
```


```python
apsan
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
      <th>업체명</th>
      <th>소재지</th>
      <th>전화번호</th>
      <th>주메뉴</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>료미(대구앞산점)</td>
      <td>대구광역시 남구 현충로 17, 2층 (대명동)</td>
      <td>0507-1400-5188</td>
      <td>경양식</td>
    </tr>
    <tr>
      <th>1</th>
      <td>올리버브라운 대구앞산카페거리점</td>
      <td>대구광역시 남구 현충로 17, 3층 (대명동)</td>
      <td>0507-1304-8192</td>
      <td>커피숍</td>
    </tr>
    <tr>
      <th>2</th>
      <td>지오네키친</td>
      <td>대구광역시 남구 현충로 21, 1층 (대명동)</td>
      <td>053-426-3992</td>
      <td>경양식</td>
    </tr>
    <tr>
      <th>3</th>
      <td>투썸플레이스대구앞산점</td>
      <td>대구광역시 남구 현충로 24 (대명동, (1~2층))</td>
      <td>053-656-9266</td>
      <td>커피숍</td>
    </tr>
    <tr>
      <th>4</th>
      <td>스타벅스대구앞산DT점</td>
      <td>대구광역시 남구 현충로 32, 1-2층 (대명동)</td>
      <td>053-623-3746</td>
      <td>커피숍</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>61</th>
      <td>파스쿠찌 대구앞산점</td>
      <td>대구광역시 남구 대명남로 191, 1,2층 (대명동)</td>
      <td>053-657-6007</td>
      <td>커피숍</td>
    </tr>
    <tr>
      <th>62</th>
      <td>도깨비나무</td>
      <td>대구광역시 남구 대명남로 193, 2층 (대명동)</td>
      <td>053-655-2231</td>
      <td>한식</td>
    </tr>
    <tr>
      <th>63</th>
      <td>오어앤</td>
      <td>대구광역시 남구 안골길 49-1, 1층 (대명동)</td>
      <td>0507-1303-9844</td>
      <td>커피숍</td>
    </tr>
    <tr>
      <th>64</th>
      <td>로더바이미(L'odeur by me)</td>
      <td>대구광역시 남구 안골길 56, 1층 (대명동)</td>
      <td>0507-1336-2821</td>
      <td>커피숍</td>
    </tr>
    <tr>
      <th>65</th>
      <td>도리도프</td>
      <td>대구광역시 남구 안골길 57, 1층 (대명동)</td>
      <td>053-655-7685</td>
      <td>경양식</td>
    </tr>
  </tbody>
</table>
<p>66 rows × 4 columns</p>
</div>



- 대구카톨릭대학교병원: 병원이라는 특수한 건물에 입지하였다. 이는 서울을 비롯한 다른지역에도 입지한 예가 많다.


```python
hospital = pd.read_excel('스타벅스매장주소(20191029).xlsx')
```


```python
hospital[hospital.지점명.str.contains('병원')]
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
      <th>지점명</th>
      <th>주소</th>
      <th>전화번호</th>
      <th>시도</th>
      <th>시군구</th>
      <th>동</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24</th>
      <td>을지병원사거리</td>
      <td>서울특별시 강남구 논현로 752 (논현동,구산빌딩)</td>
      <td>02-758-8896</td>
      <td>서울특별시</td>
      <td>강남구</td>
      <td>논현동</td>
    </tr>
    <tr>
      <th>62</th>
      <td>차병원사거리</td>
      <td>서울특별시 강남구 봉은사로 213, 센트럴타워 (논현동)</td>
      <td>02-758-8444</td>
      <td>서울특별시</td>
      <td>강남구</td>
      <td>논현동</td>
    </tr>
    <tr>
      <th>729</th>
      <td>구월길병원</td>
      <td>인천광역시 남동구 남동대로 773, , 14 삼성화재 인천사옥 (구월동)</td>
      <td>032-438-7283</td>
      <td>인천광역시</td>
      <td>남동구</td>
      <td>구월동</td>
    </tr>
    <tr>
      <th>780</th>
      <td>대전건양대병원</td>
      <td>대전광역시 서구 관저동로 170 (관저동)1층</td>
      <td>042-542-3634</td>
      <td>대전광역시</td>
      <td>서구</td>
      <td>관저동</td>
    </tr>
    <tr>
      <th>818</th>
      <td>계명대동산병원</td>
      <td>대구광역시 달서구 달구벌대로 1035 (신당동)</td>
      <td>053-584-3740</td>
      <td>대구광역시</td>
      <td>달서구</td>
      <td>신당동</td>
    </tr>
    <tr>
      <th>996</th>
      <td>삼성창원병원</td>
      <td>경상남도 창원시 마산회원구 팔용로 158 (합성동)</td>
      <td>055-253-3285</td>
      <td>경상남도</td>
      <td>창원시마산회원구</td>
      <td>합성동</td>
    </tr>
    <tr>
      <th>1063</th>
      <td>동국대일산병원</td>
      <td>경기도 고양시 일산동구 동국로 27</td>
      <td>031-961-9180</td>
      <td>경기도</td>
      <td>고양시일산동구</td>
      <td>식사동</td>
    </tr>
    <tr>
      <th>1135</th>
      <td>분당서울대병원4</td>
      <td>경기도 성남시 분당구 구미로173번길 82, 분당서울대학교병원 신관 4층 (구미동)</td>
      <td>031-716-6583</td>
      <td>경기도</td>
      <td>성남시분당구</td>
      <td>구미동</td>
    </tr>
    <tr>
      <th>1136</th>
      <td>분당서울대병원1</td>
      <td>경기도 성남시 분당구 구미로173번길 82, 분당서울대학교병원 신관 1층 (구미동)</td>
      <td>031-715-6581</td>
      <td>경기도</td>
      <td>성남시분당구</td>
      <td>구미동</td>
    </tr>
  </tbody>
</table>
</div>



-  대구율하: 주변에 아파트 단지가 형성되어있어서 도보로 이용하는것 같음


```python
ulha = pd.read_csv('대구광역시 동구_아파트 현황_20221213.csv',encoding='cp949')
```


```python
ulha[ulha.주소.str.contains('율하')]
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
      <th>연번</th>
      <th>아파트명</th>
      <th>주소</th>
      <th>동수</th>
      <th>층수</th>
      <th>세대수</th>
      <th>입주년도</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>31</td>
      <td>율하 뜨란채</td>
      <td>대구광역시 동구 율하동로 67</td>
      <td>11</td>
      <td>5</td>
      <td>332</td>
      <td>1997</td>
    </tr>
    <tr>
      <th>31</th>
      <td>32</td>
      <td>대구안심4단지</td>
      <td>대구광역시 동구 율하동로 73</td>
      <td>6</td>
      <td>5</td>
      <td>194</td>
      <td>2003</td>
    </tr>
    <tr>
      <th>32</th>
      <td>33</td>
      <td>화성아파트</td>
      <td>대구광역시 동구 율하동로20길 20</td>
      <td>9</td>
      <td>5</td>
      <td>448</td>
      <td>1990</td>
    </tr>
    <tr>
      <th>33</th>
      <td>34</td>
      <td>신기모란2차아파트</td>
      <td>대구광역시 동구 율하동로20길 30</td>
      <td>5</td>
      <td>5</td>
      <td>375</td>
      <td>1988</td>
    </tr>
    <tr>
      <th>34</th>
      <td>35</td>
      <td>신기모란3차아파트</td>
      <td>대구광역시 동구 율하동로20길 31</td>
      <td>11</td>
      <td>5</td>
      <td>640</td>
      <td>1988</td>
    </tr>
    <tr>
      <th>35</th>
      <td>36</td>
      <td>신기모란아파트</td>
      <td>대구광역시 동구 율하동로20길 40</td>
      <td>5</td>
      <td>5</td>
      <td>320</td>
      <td>1987</td>
    </tr>
    <tr>
      <th>75</th>
      <td>76</td>
      <td>안심주공 3단지</td>
      <td>대구광역시 동구 율하동로 76</td>
      <td>17</td>
      <td>6</td>
      <td>984</td>
      <td>1994</td>
    </tr>
    <tr>
      <th>116</th>
      <td>117</td>
      <td>신기메디오스</td>
      <td>대구광역시 동구 율하동로28길 54</td>
      <td>1</td>
      <td>12</td>
      <td>73</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>164</th>
      <td>165</td>
      <td>율하리버파크</td>
      <td>대구광역시 동구 율하동로 19</td>
      <td>12</td>
      <td>15</td>
      <td>486</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>165</th>
      <td>166</td>
      <td>율하휴먼시아10단지</td>
      <td>대구광역시 동구 율하동로8길 16</td>
      <td>16</td>
      <td>15</td>
      <td>976</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>166</th>
      <td>167</td>
      <td>율하 롯데캐슬 TOP LASS</td>
      <td>대구광역시 동구 율하서로 23</td>
      <td>6</td>
      <td>15</td>
      <td>447</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>167</th>
      <td>168</td>
      <td>율하휴먼시아5단지</td>
      <td>대구광역시 동구 율하서로 59</td>
      <td>12</td>
      <td>15</td>
      <td>860</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>168</th>
      <td>169</td>
      <td>e편한세상 세계육상선수촌</td>
      <td>대구광역시 동구 율하서로 85</td>
      <td>10</td>
      <td>15</td>
      <td>528</td>
      <td>2011</td>
    </tr>
  </tbody>
</table>
</div>




```python
tech = pd.read_csv('대구광역시 달성군_기업체 현황_20211112.csv',encoding='cp949')
```


```python
tech[tech.소재지주소.str.contains('테크노')]
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
      <th>업체명</th>
      <th>소재지주소</th>
      <th>생산품목</th>
      <th>대표자</th>
      <th>종업원수(명)</th>
      <th>구분</th>
      <th>위도</th>
      <th>경도</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>㈜베사</td>
      <td>대구 달성군 현풍읍 테크노대로2길 40</td>
      <td>세라믹전자부품</td>
      <td>신영철</td>
      <td>350</td>
      <td>현풍면</td>
      <td>35.686091</td>
      <td>128.447889</td>
    </tr>
    <tr>
      <th>37</th>
      <td>금오이엠에스</td>
      <td>대구 달성군 현풍읍 테크노대로4길 46</td>
      <td>버너 헤드 등</td>
      <td>구자근</td>
      <td>114</td>
      <td>현풍면</td>
      <td>35.687346</td>
      <td>128.450494</td>
    </tr>
    <tr>
      <th>40</th>
      <td>경창산업(주)</td>
      <td>대구 달성군 유가읍 테크노순환로1길 57</td>
      <td>오토 트랜스미션</td>
      <td>손일호</td>
      <td>250</td>
      <td>유가면</td>
      <td>35.677635</td>
      <td>128.447370</td>
    </tr>
    <tr>
      <th>43</th>
      <td>㈜스맥</td>
      <td>대구 달성군 유가읍 테크노대로6길 43</td>
      <td>전자응용공작기계</td>
      <td>원종범</td>
      <td>215</td>
      <td>유가면</td>
      <td>35.684904</td>
      <td>128.467021</td>
    </tr>
    <tr>
      <th>44</th>
      <td>대영코어텍㈜</td>
      <td>대구 달성군 유가읍 테크노순환로8길 8</td>
      <td>볼스크류</td>
      <td>정태호</td>
      <td>189</td>
      <td>유가면</td>
      <td>35.680113</td>
      <td>128.465760</td>
    </tr>
    <tr>
      <th>45</th>
      <td>삼금공업(주)</td>
      <td>대구 달성군 유가읍 테크노순환로8길 41</td>
      <td>자동차 엔진 부품</td>
      <td>김훈</td>
      <td>132</td>
      <td>유가면</td>
      <td>35.682167</td>
      <td>128.468042</td>
    </tr>
    <tr>
      <th>46</th>
      <td>현대로보틱스㈜</td>
      <td>대구 달성군 유가읍 테크노순환로3길 50</td>
      <td>산업용 로봇, 클린 로봇</td>
      <td>윤중근</td>
      <td>276</td>
      <td>유가면</td>
      <td>35.679718</td>
      <td>128.456354</td>
    </tr>
  </tbody>
</table>
</div>



동대구로 DT는 예전 mbc네거리에 위치하고 주변에 아파트입주예정이 많은 지역에 위치하였고, 웨딩홀과 세무서가 있다.

