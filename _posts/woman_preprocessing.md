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




    
![png](output_17_1.png)
    



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




    
![png](output_30_1.png)
    


이미지 증폭 및 저장 완료


```python

```
