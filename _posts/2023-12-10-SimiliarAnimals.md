--- 
layout: single
title: "닮은꼴 동물 예측_머신러닝"
toc: true
toc_sticky: true
toc_label: "페이지 주요 목차"

---

### 동물상 이미지 분석
## 대표적인 동물상 연예인 이미지 학습 후 닮은꼴 동물 예측

```python
import cv2, os
import matplotlib.pyplot as plt
```

### 사진 형식 jpg통일


```python
folder_names = os.listdir()
folder_names = folder_names
folder_names
```




    ['.ipynb_checkpoints', 'haerin', 'sana', 'Untitled.ipynb', 'yeji', 'yujin']




```python
file_path = folder_names[1] +'/'
```


```python
file_names = os.listdir(file_path)
```


```python
i = 1
for name in file_names:
    src = os.path.join(file_path, name)
    dst = 'w_dog_' + str(i+105) + '.jpg'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1
```

### 사이즈 바꾸고 넘파이로 저장


```python
import numpy as np
from rembg import remove
```


```python
folder_path = ['dog','cat','rabbit','snake']
```


```python
file_path = './' + folder_path[0]
```


```python
file_names = os.listdir(file_path)
```


```python
dsize_ = (100,100)
dogList = []
for name in file_names:
    #파일 읽기    
    src = os.path.join(file_path, name)    
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)

    # 파일 사이즈 바꾸기
    org = cv2.resize(img, dsize_)
    
    # 파일 돌리기
    img = remove(org)
    
    dogList.append(img)

```


```python
np.save('dog',dogList)
```


```python
file_path = './' + folder_path[1]
file_names = os.listdir(file_path)
```


```python
dsize_ = (100,100)
catList = []
for name in file_names:
    #파일 읽기    
    src = os.path.join(file_path, name)    
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)

    # 파일 사이즈 바꾸기
    org = cv2.resize(img, dsize_)
    
    # 파일 돌리기
    img = remove(org)
    
    catList.append(img)
np.save('cat',catList)
```


```python
len(dogList), len(catList)
```




    (247, 219)




```python
catList[0].shape
```




    (100, 100, 4)




```python
plt.imshow(dogList[10])
```




    <matplotlib.image.AxesImage at 0x2040dfe7e50>




    
![output_17_1](https://github.com/2Seungsu/AI-BigData_curriculum/assets/141051562/6f828692-86fe-45d9-b95d-1e1a6641bcc8)    



```python
file_path = './' + folder_path[2]
file_names = os.listdir(file_path)
```


```python
dsize_ = (100,100)
rabbitList = []
for name in file_names:
    try:    #파일 읽기    
        src = os.path.join(file_path, name)    
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    
        # 파일 사이즈 바꾸기
        org = cv2.resize(img, dsize_)
        
        # 파일 돌리기
        img = remove(org)
        rabbitList.append(img)
    except Exception:
        continue
np.save('rabbit',rabbitList)
```


```python
file_path = './' + folder_path[3]
file_names = os.listdir(file_path)
```


```python
dsize_ = (100,100)
snakeList = []
for name in file_names:
    try :
        #파일 읽기    
        src = os.path.join(file_path, name)    
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    
        # 파일 사이즈 바꾸기
        org = cv2.resize(img, dsize_)
        
        # 파일 돌리기
        img = remove(org)
        snakeList.append(img)
    except Exception:
        continue
np.save('snake',snakeList)
```

### 흑백 및 각도 조절 검증


```python
height, width, ch = dogList[10].shape
```


```python
dogList[14].shape
```




    (100, 100, 4)




```python
dogDegree = []
```


```python
dogDegree = []

for i in dogList:
    
    for degree in range(0,360,10):
        # 변환행렬 만들기
        matrix = cv2.getRotationMatrix2D((width//2, height//2), degree, 1)  
                                         #    회전 중심좌표     각도   원본배율
        
        #회전 변환 행렬에 따른 회전 이미지 반환
        dst = cv2.warpAffine(i, matrix,   (width, height))
                            #   원본                  변환행렬     너비높이 
        dogDegree.append(dst)
```


```python
len(dogDegree)
```




    8892




```python
catDegree = []

for i in catList:
    
    for degree in range(0,360,10):
        # 변환행렬 만들기
        matrix = cv2.getRotationMatrix2D((width//2, height//2), degree, 1)  
                                         #    회전 중심좌표     각도   원본배율
        
        #회전 변환 행렬에 따른 회전 이미지 반환
        dst = cv2.warpAffine(i, matrix,   (width, height))
                            #   원본                  변환행렬     너비높이 
        catDegree.append(dst)
```


```python
catDegree[0].shape
```




    (100, 100, 4)




```python
plt.imshow(catDegree[5000])
```




    <matplotlib.image.AxesImage at 0x2040bded1f0>




    
![output_30_1](https://github.com/2Seungsu/AI-BigData_curriculum/assets/141051562/d153fe61-e115-4a88-86dc-6f5ff70f6821)

    


이미지 증폭 및 저장 완료


### 고양이, 개, 토끼 ,뱀 닮은 연예인 numpydata


```python
catList = np.load('cat.npy')
dogList = np.load('dog.npy')
```


```python
rabbitList = np.load('rabbit.npy')
snakeList = np.load('snake.npy')
```


```python
rabbitList[0].shape, dogList[10].shape,snakeList[0].shape
```




    ((100, 100, 4), (100, 100, 4), (100, 100, 4))




```python
height, width, ch = dogList[10].shape
```


```python
dogDegree = []

for i in dogList:
    
    for degree in range(0,360,10):
        # 변환행렬 만들기
        matrix = cv2.getRotationMatrix2D((width//2, height//2), degree, 1)  
                                         #    회전 중심좌표     각도   원본배율
        
        #회전 변환 행렬에 따른 회전 이미지 반환
        dst = cv2.warpAffine(i, matrix,   (width, height))
                            #   원본                  변환행렬     너비높이 
        dogDegree.append(dst)
```


```python
catDegree = []

for i in catList:
    
    for degree in range(0,360,10):
        # 변환행렬 만들기
        matrix = cv2.getRotationMatrix2D((width//2, height//2), degree, 1)  
                                         #    회전 중심좌표     각도   원본배율
        
        #회전 변환 행렬에 따른 회전 이미지 반환
        dst = cv2.warpAffine(i, matrix,   (width, height))
                            #   원본                  변환행렬     너비높이 
        catDegree.append(dst)
```


```python
dogDegree255=[i/255 for i in dogDegree]
catDegree255=[i/255 for i in catDegree]
dogDegree255 = [i.flatten().astype(np.uint8) for i in dogDegree255]
catDegree255 = [i.flatten().astype(np.int8) for i in catDegree255]

```


```python
rabbitDegree = []

for i in rabbitList:
    
    for degree in range(0,360,10):
        # 변환행렬 만들기
        matrix = cv2.getRotationMatrix2D((width//2, height//2), degree, 1)  
                                         #    회전 중심좌표     각도   원본배율
        
        #회전 변환 행렬에 따른 회전 이미지 반환
        dst = cv2.warpAffine(i, matrix,   (width, height))
                            #   원본                  변환행렬     너비높이 
        rabbitDegree.append(dst)
```


```python
snakeDegree = []

for i in snakeList:
    
    for degree in range(0,360,10):
        # 변환행렬 만들기
        matrix = cv2.getRotationMatrix2D((width//2, height//2), degree, 1)  
                                         #    회전 중심좌표     각도   원본배율
        
        #회전 변환 행렬에 따른 회전 이미지 반환
        dst = cv2.warpAffine(i, matrix,   (width, height))
                            #   원본                  변환행렬     너비높이 
        snakeDegree.append(dst)
```

### 정규화


```python
rabbitDegree255=[i/255 for i in rabbitDegree]
snakeDegree255=[i/255 for i in snakeDegree]
rabbitDegree255 = [i.flatten().astype(np.uint8) for i in rabbitDegree255]
snakeDegree255 = [i.flatten().astype(np.uint8) for i in snakeDegree255]
```


```python
#정규화한거 합치기
feature_snake=np.vstack(snakeDegree255)
feature_rabbit=np.vstack(rabbitDegree255)
feature_cat = np.vstack(catDegree255)
feature_dog = np.vstack(dogDegree255)
```


```python
len(feature_snake), len(feature_rabbit), len(feature_cat), len(feature_dog)
```




    (6228, 11772, 7884, 8892)




```python
target_snake=np.array([3,]*len(feature_snake))
target_rabbit=np.array([2,]*len(feature_rabbit))
target_dog=np.array([1,]*len(feature_dog))
target_cat=np.array([0,]*len(feature_cat))
```

### 피처


```python
feature=np.vstack((feature_cat,feature_dog,feature_rabbit,feature_snake))
```


```python
feature.shape
```




    (34776, 40000)



### 타겟


```python
target = np.concatenate((target_cat,target_dog,target_rabbit,target_snake), axis=0)
```


```python
target.shape
```




    (34776,)




```python
np.save('feature_snake',feature_snake)
np.save('feature_rabbit', feature_rabbit)
np.save('feature_dog', feature_dog)
np.save('feature_cat',feature_cat)
np.save('target',target)
```

### 트레인 테스트 나누기


```python
from sklearn.model_selection import train_test_split
```


```python
trainX, valX, trainy, valy = train_test_split(feature, target,
                                             stratify=target,
                                             random_state=85)
```

### 모델학습


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
r1 =  RandomForestClassifier()
```


```python
r1.fit(trainX,trainy)
```

### 모델저장


```python
import joblib
```


```python
model_file = 'woman_animal.pkl'
joblib.dump(r1, model_file)
```




    ['woman_animal.pkl']




```python
r1.score(trainX,trainy), r1.score(valX,valy)
```




    (0.9587838355954298, 0.3493213710605015)





