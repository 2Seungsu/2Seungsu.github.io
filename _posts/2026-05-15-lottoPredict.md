--- 
layout: single
title: "예측 - 로또 번호 추천"
toc: true
toc_sticky: true
toc_label: "페이지 주요 목차"

---

### 2002~2026년 로또 1등 번호 추천  
## 2002~2026년 로또 1등 번호 패턴, 조합을 학습한 AI모델

<br><br><br><br><br>

### 데이터 불러오기


```python
import pandas as pd 
import numpy as np
from glob import glob
from tqdm.auto import tqdm
```


```python
lst = glob('./history/*')
dfs = []
for i in tqdm(lst):
    dfs.append(pd.read_csv(i))
for i in range(2002,2027):
    globals()[f'df_{i}'] = dfs[i-2002]
    globals()[f'df_{i}']['year'] = i
```


      0%|          | 0/25 [00:00<?, ?it/s]


### 2019년도부터 티비 방송 종료
### 이전년 것도 학습하지만 2019년부터 현재까지 학습률 강화


```python
df = pd.concat(dfs)
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>bonus</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14</td>
      <td>27</td>
      <td>30</td>
      <td>31</td>
      <td>40</td>
      <td>42</td>
      <td>2</td>
      <td>2002</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11</td>
      <td>16</td>
      <td>19</td>
      <td>21</td>
      <td>27</td>
      <td>31</td>
      <td>30</td>
      <td>2002</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>13</td>
      <td>21</td>
      <td>25</td>
      <td>32</td>
      <td>42</td>
      <td>2</td>
      <td>2002</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>23</td>
      <td>29</td>
      <td>33</td>
      <td>37</td>
      <td>40</td>
      <td>16</td>
      <td>2002</td>
    </tr>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>14</td>
      <td>30</td>
      <td>31</td>
      <td>33</td>
      <td>37</td>
      <td>19</td>
      <td>2003</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2</td>
      <td>17</td>
      <td>20</td>
      <td>35</td>
      <td>37</td>
      <td>39</td>
      <td>24</td>
      <td>2026</td>
    </tr>
    <tr>
      <th>17</th>
      <td>6</td>
      <td>27</td>
      <td>30</td>
      <td>36</td>
      <td>38</td>
      <td>42</td>
      <td>25</td>
      <td>2026</td>
    </tr>
    <tr>
      <th>18</th>
      <td>10</td>
      <td>22</td>
      <td>24</td>
      <td>27</td>
      <td>38</td>
      <td>45</td>
      <td>11</td>
      <td>2026</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>3</td>
      <td>17</td>
      <td>26</td>
      <td>27</td>
      <td>42</td>
      <td>23</td>
      <td>2026</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>4</td>
      <td>16</td>
      <td>23</td>
      <td>31</td>
      <td>41</td>
      <td>2</td>
      <td>2026</td>
    </tr>
  </tbody>
</table>
<p>1225 rows × 8 columns</p>
</div>


#### 피쳐엔지니어링

```python
import random
from itertools import combinations
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 최최근 회차 번호
prev_nums = df.iloc[-2][['1','2','3','4','5','6']].tolist()

# 최최최근까지 학습데이터로 사용
df = df.iloc[:-2]

# =========================================
# Feature 생성 함수
# =========================================

def make_features(nums, prev_nums):

    nums = sorted(nums)

    total = sum(nums)

    odd = len([n for n in nums if n % 2 == 1])
    even = 6 - odd

    low = len([n for n in nums if n <= 22])
    high = 6 - low

    # 연번 개수
    consecutive = 0

    for i in range(5):
        if nums[i+1] - nums[i] == 1:
            consecutive += 1

    # 직전 회차 겹침
    overlap = len(set(nums) & set(prev_nums))

    # 끝수 중복
    tails = [n % 10 for n in nums]
    same_tail = 1 if len(tails) != len(set(tails)) else 0


    MAX_LOOP = 30

    
    # 최근 30회동안 번호가 얼마나 나왔는지
    recent = df.tail(30)
    # 전체 번호 펼치기
    recent_numbers = recent[number_cols].values.flatten()
    # 각 번호 출현 횟수
    freq_dict = pd.Series(recent_numbers).value_counts().to_dict()
    # 현재 조합 번호들의 출현 횟수
    freqs = [freq_dict.get(n, 0) for n in nums]


    # 번호가 안나온지 몇회차나 지났는지
    # 최신 회차부터 역순 탐색
    MAX_LOOP = 30    
    skips = []
    reversed_df = df.iloc[::-2]    
    for n in nums:
        skip = 0
        found = False
        loop_count = 0
        for _, row in reversed_df.iterrows():
            if loop_count >= MAX_LOOP:
                break
            row_nums = row[number_cols].tolist()
            if n in row_nums:
                found = True
                break
            skip += 1
            loop_count += 1
        if not found:
            skip = min(len(df), MAX_LOOP)
        skips.append(skip)


    # 과거 현재 조합이 얼마나 나왔는지
    # 현재 조합의 pair 생성
    MAX_LOOP = 30
    current_pairs = list(combinations(nums, 2))
    pair_counts = []
    for pair in current_pairs:
        count = 0
        a, b = pair
        loop_count = 0
        for _, row in df.iterrows():
            # 300번 넘으면 중단
            if loop_count >= MAX_LOOP:
                break
            row_nums = row[number_cols].tolist()
            if a in row_nums and b in row_nums:
                count += 1
            loop_count += 1
        pair_counts.append(count)

    

    features = {

        "sum": total,

        "odd": odd,
        "even": even,

        "low": low,
        "high": high,

        "consecutive": consecutive,

        "overlap_prev": overlap,

        "same_tail": same_tail,

        "min": min(nums),
        "max": max(nums),

        "recent_freq_mean": np.mean(freqs),
        "skip_mean": np.mean(skips),
        "pair_freq_mean": np.mean(pair_counts),
    }

    return features
```


```python
# =========================================
# 라벨 생성 함수
# =========================================
def make_label(features):

    score = 0

    # 번호합
    if 110 <= features['sum'] <= 170:
        score += 1

    # 홀짝
    if features['odd'] in [2,3,4]:
        score += 1

    # 저고
    if features['low'] in [2,3,4]:
        score += 1

    # 연번
    if features['consecutive'] <= 1:
        score += 1

    # 직전 번호 중복
    if features['overlap_prev'] == 1:
        score += 1

    # 끝수
    if features['same_tail'] == 1:
        score += 1

    # 최근 30회 동안 평균 출현 횟수
    if features['recent_freq_mean'] >= 3 and features['recent_freq_mean'] <= 5:
        score += 1
    # 최근 30회 동안 평균 스킵 횟수
    if features['skip_mean'] >= 1 and features['skip_mean'] <= 9:
        score += 1
    # 최근 30회 동안 pair 횟수
    if features['pair_freq_mean'] >= 0.3:
        score += 1    

    # 조건 많이 만족하면 1
    return 1 if score >= 7 else 0
```

#### 데이터셋 생성
```python
# =========================================
# 학습 데이터 생성
# =========================================
X = []
y = []
sample_weights = []

# 당첨 1 (1,2등 번호가 당첨으로 가정)
number_cols = ['1', '2', '3', '4', '5', '6']
for i in tqdm(range(1, len(df))):
    # 연도
    year = df.iloc[i]['year']
    if 2019 <= year <= 2026:
        weight = 2.0
    else:
        weight = 1.0

    # 전 회차 1등번호 랜덤 추출
    nums = df.iloc[i][number_cols].tolist()    
    features = make_features(nums, prev_nums)
    # print(make_label(features))
    # make_label 기준 넘는거만 피쳐로 생성
    if make_label(features) == 1:
        prev_nums = df.iloc[i-1][number_cols].tolist()        
        X.append(list(features.values()))
        y.append(1)
        sample_weights.append(weight)   

# 비당첨 0
for i in tqdm(range(1, len(df))):
    prev_nums = df.iloc[i-1][number_cols].tolist()        
    # 당첨 데이터보다 랜덤 데이터가 더 많아야 모델이 학습 잘함.
    for _ in range(2):
        nums = sorted(random.sample(range(1,46),6))
        features = make_features(nums, prev_nums)
        X.append(list(features.values()))
        y.append(0)
        sample_weights.append(1)

# 학습 데이터셋 생성
feature_names = list(features.keys())
X = pd.DataFrame(X, columns=feature_names)
y = np.array(y)
```


      0%|          | 0/1222 [00:00<?, ?it/s]



      0%|          | 0/1222 [00:00<?, ?it/s]



```python
# =========================================
# train/test split
# =========================================
X_train, X_test, y_train, y_test, sample_weights_train, sample_weights_test = train_test_split(
    X,
    y,
    sample_weights,
    test_size=0.3,
    random_state=42,
    shuffle=False  # 시계열 데이터에서 과거로 미래를 예측하기 때문에 순서가 섞이면 안됨
)
```


```python
X_train.shape, y_train.shape
```




    ((2271, 13), (2271,))




```python
X_test.shape, y_test.shape
```




    ((974, 13), (974,))




```python
len(sample_weights), len(sample_weights_train), len(sample_weights_test)
```




    (3245, 2271, 974)


#### 모델 학습

```python
# =========================================
# LightGBM 모델
# =========================================

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=5,
    num_leaves=15,
    random_state=42,
    force_col_wise=True,
    min_data_in_leaf = 10
)

# 학습
model.fit(X_train, y_train, sample_weight=sample_weights_train)
# 예측
pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)

```

    
    
    [LightGBM] [Info] Number of positive: 801, number of negative: 1470
    [LightGBM] [Info] Total Bins 341
    [LightGBM] [Info] Number of data points in the train set: 2271, number of used features: 13
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.417822 -> initscore=-0.331722
    [LightGBM] [Info] Start training from score -0.331722
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    
    Accuracy: 0.704312114989733
    


```python
# 모델 저장
import joblib
joblib.dump(model,'lotto_pred.joblib')
# 불러오기
#model = joblib.load('lotto_pred.joblib')
```




    ['lotto_pred.joblib']




```python
# =========================================
# 추천 번호 생성
# =========================================
results = []
# 1,2,3 등에 걸릴확률 약 1/35000
for _ in tqdm(range(35000)):
# for _ in tqdm(range(10)):
    # 최근 회차 번호로 이번거 예측
    latest_prev = df.iloc[-1][['1','2','3','4','5','6']].tolist()
    nums = sorted(random.sample(range(1,46),6))
    features = make_features(nums, latest_prev)

    x = pd.DataFrame(
        [list(features.values())],
        columns=feature_names
    )
    prob = model.predict_proba(x)[0][1]
    results.append((nums, prob))

# 점수 높은 순 정렬
results = sorted(results, key=lambda x: x[1], reverse=True)

# TOP 10 출력
print("\n추천 번호 TOP10\n")

for nums, prob in results[:10]:
    print(nums, round(prob, 4))
```


      0%|          | 0/20000 [00:00<?, ?it/s]


    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf    

    
    
    추천 번호 TOP10
    
    [10, 17, 26, 37, 40, 43] 0.9542
    [10, 20, 27, 33, 37, 38] 0.9538
    [10, 14, 33, 37, 39, 44] 0.9497
    [10, 14, 16, 27, 30, 37] 0.9485
    [7, 9, 18, 21, 23, 35] 0.9432
    [10, 18, 23, 29, 37, 40] 0.9408
    [3, 10, 33, 35, 37, 44] 0.9358
    [5, 10, 27, 33, 37, 40] 0.9352
    [7, 13, 27, 30, 33, 40] 0.935
    [7, 10, 19, 21, 27, 33] 0.9346
    
