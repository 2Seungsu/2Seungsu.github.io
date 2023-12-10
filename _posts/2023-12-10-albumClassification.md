--- 
layout: single
title: "딥러닝 - 앨범 이미지 시대별 분류"
toc: true
toc_sticky: true
toc_label: "페이지 주요 목차"

---

### 앨범 이미지 시대별 특징 추정
## 앨범 이미지를 CNN 딥러닝 모델로 학습 후 시대별 분류 및 예측 

```python
import numpy as np
```


```python
from google.colab import drive
drive.mount('/content/drive')
```


```python
feature = np.load('/content/drive/MyDrive/feature.npy')
target = np.load('/content/drive/MyDrive/target.npy')
```


```python
feature.shape, target.shape
```




    ((269640, 48, 48, 3), (269640, 3))



### 모델 컴파일


```python
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.applications import VGG16
```


```python
transfer_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
transfer_model.trainable=False
transfer_model.summary()
```

    Model: "vgg16"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_1 (InputLayer)        [(None, 48, 48, 3)]       0         
                                                                     
     block1_conv1 (Conv2D)       (None, 48, 48, 64)        1792      
                                                                     
     block1_conv2 (Conv2D)       (None, 48, 48, 64)        36928     
                                                                     
     block1_pool (MaxPooling2D)  (None, 24, 24, 64)        0         
                                                                     
     block2_conv1 (Conv2D)       (None, 24, 24, 128)       73856     
                                                                     
     block2_conv2 (Conv2D)       (None, 24, 24, 128)       147584    
                                                                     
     block2_pool (MaxPooling2D)  (None, 12, 12, 128)       0         
                                                                     
     block3_conv1 (Conv2D)       (None, 12, 12, 256)       295168    
                                                                     
     block3_conv2 (Conv2D)       (None, 12, 12, 256)       590080    
                                                                     
     block3_conv3 (Conv2D)       (None, 12, 12, 256)       590080    
                                                                     
     block3_pool (MaxPooling2D)  (None, 6, 6, 256)         0         
                                                                     
     block4_conv1 (Conv2D)       (None, 6, 6, 512)         1180160   
                                                                     
     block4_conv2 (Conv2D)       (None, 6, 6, 512)         2359808   
                                                                     
     block4_conv3 (Conv2D)       (None, 6, 6, 512)         2359808   
                                                                     
     block4_pool (MaxPooling2D)  (None, 3, 3, 512)         0         
                                                                     
     block5_conv1 (Conv2D)       (None, 3, 3, 512)         2359808   
                                                                     
     block5_conv2 (Conv2D)       (None, 3, 3, 512)         2359808   
                                                                     
     block5_conv3 (Conv2D)       (None, 3, 3, 512)         2359808   
                                                                     
     block5_pool (MaxPooling2D)  (None, 1, 1, 512)         0         
                                                                     
    =================================================================
    Total params: 14714688 (56.13 MB)
    Trainable params: 0 (0.00 Byte)
    Non-trainable params: 14714688 (56.13 MB)
    _________________________________________________________________
    


```python
model = Sequential()
model.add(transfer_model)
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     vgg16 (Functional)          (None, 1, 1, 512)         14714688  
                                                                     
     flatten (Flatten)           (None, 512)               0         
                                                                     
     dense (Dense)               (None, 64)                32832     
                                                                     
     activation (Activation)     (None, 64)                0         
                                                                     
     dropout (Dropout)           (None, 64)                0         
                                                                     
     dense_1 (Dense)             (None, 3)                 195       
                                                                     
     activation_1 (Activation)   (None, 3)                 0         
                                                                     
    =================================================================
    Total params: 14747715 (56.26 MB)
    Trainable params: 33027 (129.01 KB)
    Non-trainable params: 14714688 (56.13 MB)
    _________________________________________________________________
    


```python
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0002), metrics='accuracy')
```


```python
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(feature, target,
                                                    test_size=0.2,
                                                    random_state=12,
                                                    stratify=target)
```

### 하이퍼파라미터 튜닝


```python
from kerastuner.tuners import RandomSearch
```

    <ipython-input-12-94471a811b41>:1: DeprecationWarning: `import kerastuner` is deprecated, please use `import keras_tuner`.
      from kerastuner.tuners import RandomSearch
    


```python
def build_model(hp):
    model = Sequential()
    model.add(Flatten(input_shape=(48,48,3)))

    # 하이퍼파라미터로 조정할 레이어 수 및 유닛 수
    for i in range(hp.Int('num_layers', min_value=2, max_value=20)):
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
                               activation='relu'))

    model.add(Dense(3, activation='softmax'))

    # 옵티마이저 및 학습률 설정
    hp_optimizer = hp.Choice('optimizer', values=['adam', 'sgd'])
    hp_learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    if hp_optimizer == 'adam':
        optimizer = optimizers.Adam(learning_rate=hp_learning_rate)
    else:
        optimizer = optimizers.SGD(learning_rate=hp_learning_rate)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

```


```python
# 하이퍼파라미터 튜닝을 위한 튜너 설정 (RandomSearch 사용)
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # 탐색할 하이퍼파라미터 조합 수
    directory='my_tuner_dir',  # 결과를 저장할 디렉토리
    project_name='melon_tuning'  # 프로젝트 이름
)
```


```python
# 튜닝 수행
tuner.search(train_X, train_y, epochs=5, batch_size=1024, validation_split=0.2)

# 최적의 하이퍼파라미터 조합 출력
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best Hyperparameters: {best_hps}")

# 최적의 모델 가져오기
best_model = tuner.get_best_models(num_models=1)[0]

# 최적의 모델 훈련
best_model.fit(train_X, train_y, epochs=100, batch_size=1024, validation_split=0.2)
```

    Trial 5 Complete [00h 00m 25s]
    val_accuracy: 0.45993557572364807
    
    Best val_accuracy So Far: 0.45993557572364807
    Total elapsed time: 00h 01m 36s
    Best Hyperparameters: <keras_tuner.engine.hyperparameters.hyperparameters.HyperParameters object at 0x7d5e455e82b0>
    Epoch 1/100
    169/169 [==============================] - 9s 25ms/step - loss: 0.9899 - accuracy: 0.4838 - val_loss: 0.9941 - val_accuracy: 0.4834
    Epoch 2/100
    169/169 [==============================] - 3s 18ms/step - loss: 0.9702 - accuracy: 0.4990 - val_loss: 0.9630 - val_accuracy: 0.5103
    Epoch 3/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.9450 - accuracy: 0.5190 - val_loss: 0.9432 - val_accuracy: 0.5224
    Epoch 4/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.9272 - accuracy: 0.5341 - val_loss: 0.9321 - val_accuracy: 0.5310
    Epoch 5/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.9014 - accuracy: 0.5519 - val_loss: 0.9126 - val_accuracy: 0.5458
    Epoch 6/100
    169/169 [==============================] - 3s 16ms/step - loss: 0.8795 - accuracy: 0.5663 - val_loss: 0.9105 - val_accuracy: 0.5547
    Epoch 7/100
    169/169 [==============================] - 3s 17ms/step - loss: 0.8520 - accuracy: 0.5859 - val_loss: 0.9014 - val_accuracy: 0.5569
    Epoch 8/100
    169/169 [==============================] - 3s 16ms/step - loss: 0.8317 - accuracy: 0.5993 - val_loss: 0.8599 - val_accuracy: 0.5856
    Epoch 9/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.8021 - accuracy: 0.6181 - val_loss: 0.8501 - val_accuracy: 0.5947
    Epoch 10/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.7837 - accuracy: 0.6296 - val_loss: 0.8464 - val_accuracy: 0.6019
    Epoch 11/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.7638 - accuracy: 0.6401 - val_loss: 0.8258 - val_accuracy: 0.6120
    Epoch 12/100
    169/169 [==============================] - 3s 15ms/step - loss: 0.7377 - accuracy: 0.6552 - val_loss: 0.8367 - val_accuracy: 0.6093
    Epoch 13/100
    169/169 [==============================] - 3s 18ms/step - loss: 0.7253 - accuracy: 0.6613 - val_loss: 0.8211 - val_accuracy: 0.6228
    Epoch 14/100
    169/169 [==============================] - 3s 15ms/step - loss: 0.6944 - accuracy: 0.6797 - val_loss: 0.7927 - val_accuracy: 0.6388
    Epoch 15/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.6804 - accuracy: 0.6861 - val_loss: 0.8485 - val_accuracy: 0.6169
    Epoch 16/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.6587 - accuracy: 0.6990 - val_loss: 0.7895 - val_accuracy: 0.6481
    Epoch 17/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.6459 - accuracy: 0.7053 - val_loss: 0.7836 - val_accuracy: 0.6511
    Epoch 18/100
    169/169 [==============================] - 3s 17ms/step - loss: 0.6354 - accuracy: 0.7100 - val_loss: 0.7718 - val_accuracy: 0.6589
    Epoch 19/100
    169/169 [==============================] - 3s 20ms/step - loss: 0.6141 - accuracy: 0.7213 - val_loss: 0.7697 - val_accuracy: 0.6671
    Epoch 20/100
    169/169 [==============================] - 2s 15ms/step - loss: 0.6037 - accuracy: 0.7269 - val_loss: 0.7997 - val_accuracy: 0.6562
    Epoch 21/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.5857 - accuracy: 0.7365 - val_loss: 0.7891 - val_accuracy: 0.6657
    Epoch 22/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.5728 - accuracy: 0.7424 - val_loss: 0.7843 - val_accuracy: 0.6781
    Epoch 23/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.5655 - accuracy: 0.7473 - val_loss: 0.7885 - val_accuracy: 0.6802
    Epoch 24/100
    169/169 [==============================] - 3s 19ms/step - loss: 0.5483 - accuracy: 0.7560 - val_loss: 0.7724 - val_accuracy: 0.6823
    Epoch 25/100
    169/169 [==============================] - 3s 16ms/step - loss: 0.5450 - accuracy: 0.7571 - val_loss: 0.7642 - val_accuracy: 0.6844
    Epoch 26/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.5248 - accuracy: 0.7677 - val_loss: 0.7833 - val_accuracy: 0.6861
    Epoch 27/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.5168 - accuracy: 0.7720 - val_loss: 0.8090 - val_accuracy: 0.6818
    Epoch 28/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.5092 - accuracy: 0.7741 - val_loss: 0.8026 - val_accuracy: 0.6870
    Epoch 29/100
    169/169 [==============================] - 3s 17ms/step - loss: 0.4992 - accuracy: 0.7801 - val_loss: 0.7837 - val_accuracy: 0.7024
    Epoch 30/100
    169/169 [==============================] - 3s 19ms/step - loss: 0.4912 - accuracy: 0.7831 - val_loss: 0.8181 - val_accuracy: 0.6810
    Epoch 31/100
    169/169 [==============================] - 2s 15ms/step - loss: 0.4814 - accuracy: 0.7884 - val_loss: 0.8007 - val_accuracy: 0.6987
    Epoch 32/100
    169/169 [==============================] - 3s 15ms/step - loss: 0.4741 - accuracy: 0.7915 - val_loss: 0.8168 - val_accuracy: 0.6961
    Epoch 33/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.4591 - accuracy: 0.7980 - val_loss: 0.8415 - val_accuracy: 0.6973
    Epoch 34/100
    169/169 [==============================] - 3s 15ms/step - loss: 0.4600 - accuracy: 0.7974 - val_loss: 0.7841 - val_accuracy: 0.7080
    Epoch 35/100
    169/169 [==============================] - 3s 20ms/step - loss: 0.4443 - accuracy: 0.8057 - val_loss: 0.8191 - val_accuracy: 0.6918
    Epoch 36/100
    169/169 [==============================] - 3s 17ms/step - loss: 0.4431 - accuracy: 0.8063 - val_loss: 0.8481 - val_accuracy: 0.6996
    Epoch 37/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.4352 - accuracy: 0.8102 - val_loss: 0.8325 - val_accuracy: 0.7082
    Epoch 38/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.4211 - accuracy: 0.8162 - val_loss: 0.8323 - val_accuracy: 0.7131
    Epoch 39/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.4225 - accuracy: 0.8165 - val_loss: 0.8720 - val_accuracy: 0.6872
    Epoch 40/100
    169/169 [==============================] - 3s 17ms/step - loss: 0.4175 - accuracy: 0.8175 - val_loss: 0.8389 - val_accuracy: 0.7043
    Epoch 41/100
    169/169 [==============================] - 3s 18ms/step - loss: 0.4087 - accuracy: 0.8228 - val_loss: 0.8776 - val_accuracy: 0.7066
    Epoch 42/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.4029 - accuracy: 0.8250 - val_loss: 0.8504 - val_accuracy: 0.7123
    Epoch 43/100
    169/169 [==============================] - 3s 15ms/step - loss: 0.3973 - accuracy: 0.8280 - val_loss: 0.9169 - val_accuracy: 0.7053
    Epoch 44/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.3907 - accuracy: 0.8294 - val_loss: 0.8993 - val_accuracy: 0.7165
    Epoch 45/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.3976 - accuracy: 0.8278 - val_loss: 0.8648 - val_accuracy: 0.7200
    Epoch 46/100
    169/169 [==============================] - 3s 19ms/step - loss: 0.3726 - accuracy: 0.8393 - val_loss: 0.9060 - val_accuracy: 0.7194
    Epoch 47/100
    169/169 [==============================] - 3s 17ms/step - loss: 0.3684 - accuracy: 0.8408 - val_loss: 0.8914 - val_accuracy: 0.7145
    Epoch 48/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.3648 - accuracy: 0.8425 - val_loss: 0.9112 - val_accuracy: 0.7195
    Epoch 49/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.3666 - accuracy: 0.8421 - val_loss: 0.9079 - val_accuracy: 0.7162
    Epoch 50/100
    169/169 [==============================] - 4s 23ms/step - loss: 0.3697 - accuracy: 0.8400 - val_loss: 0.9430 - val_accuracy: 0.7181
    Epoch 51/100
    169/169 [==============================] - 5s 28ms/step - loss: 0.3530 - accuracy: 0.8483 - val_loss: 0.9301 - val_accuracy: 0.7251
    Epoch 52/100
    169/169 [==============================] - 5s 27ms/step - loss: 0.3490 - accuracy: 0.8501 - val_loss: 0.9310 - val_accuracy: 0.7120
    Epoch 53/100
    169/169 [==============================] - 4s 25ms/step - loss: 0.3474 - accuracy: 0.8513 - val_loss: 0.9038 - val_accuracy: 0.7215
    Epoch 54/100
    169/169 [==============================] - 4s 24ms/step - loss: 0.3476 - accuracy: 0.8509 - val_loss: 0.9294 - val_accuracy: 0.7233
    Epoch 55/100
    169/169 [==============================] - 3s 18ms/step - loss: 0.3327 - accuracy: 0.8576 - val_loss: 0.9431 - val_accuracy: 0.7276
    Epoch 56/100
    169/169 [==============================] - 3s 15ms/step - loss: 0.3369 - accuracy: 0.8558 - val_loss: 0.9532 - val_accuracy: 0.7198
    Epoch 57/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.3290 - accuracy: 0.8595 - val_loss: 1.0017 - val_accuracy: 0.7309
    Epoch 58/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.3194 - accuracy: 0.8635 - val_loss: 0.9848 - val_accuracy: 0.7222
    Epoch 59/100
    169/169 [==============================] - 2s 15ms/step - loss: 0.3310 - accuracy: 0.8586 - val_loss: 0.9540 - val_accuracy: 0.7223
    Epoch 60/100
    169/169 [==============================] - 3s 19ms/step - loss: 0.3198 - accuracy: 0.8628 - val_loss: 0.9380 - val_accuracy: 0.7238
    Epoch 61/100
    169/169 [==============================] - 3s 18ms/step - loss: 0.3095 - accuracy: 0.8683 - val_loss: 0.9423 - val_accuracy: 0.7193
    Epoch 62/100
    169/169 [==============================] - 2s 15ms/step - loss: 0.3250 - accuracy: 0.8615 - val_loss: 0.9251 - val_accuracy: 0.7284
    Epoch 63/100
    169/169 [==============================] - 3s 16ms/step - loss: 0.3048 - accuracy: 0.8698 - val_loss: 0.9927 - val_accuracy: 0.7213
    Epoch 64/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.3017 - accuracy: 0.8714 - val_loss: 0.9862 - val_accuracy: 0.7266
    Epoch 65/100
    169/169 [==============================] - 3s 17ms/step - loss: 0.3047 - accuracy: 0.8702 - val_loss: 0.9892 - val_accuracy: 0.7211
    Epoch 66/100
    169/169 [==============================] - 3s 17ms/step - loss: 0.2986 - accuracy: 0.8726 - val_loss: 0.9826 - val_accuracy: 0.7315
    Epoch 67/100
    169/169 [==============================] - 3s 15ms/step - loss: 0.3042 - accuracy: 0.8709 - val_loss: 0.9989 - val_accuracy: 0.7243
    Epoch 68/100
    169/169 [==============================] - 2s 15ms/step - loss: 0.2827 - accuracy: 0.8801 - val_loss: 1.0096 - val_accuracy: 0.7346
    Epoch 69/100
    169/169 [==============================] - 2s 15ms/step - loss: 0.2894 - accuracy: 0.8770 - val_loss: 1.0531 - val_accuracy: 0.7276
    Epoch 70/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.2940 - accuracy: 0.8752 - val_loss: 1.0053 - val_accuracy: 0.7318
    Epoch 71/100
    169/169 [==============================] - 3s 17ms/step - loss: 0.2962 - accuracy: 0.8741 - val_loss: 1.0239 - val_accuracy: 0.7333
    Epoch 72/100
    169/169 [==============================] - 3s 18ms/step - loss: 0.2776 - accuracy: 0.8823 - val_loss: 1.0369 - val_accuracy: 0.7300
    Epoch 73/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.2736 - accuracy: 0.8852 - val_loss: 1.0519 - val_accuracy: 0.7295
    Epoch 74/100
    169/169 [==============================] - 2s 15ms/step - loss: 0.2747 - accuracy: 0.8835 - val_loss: 1.0669 - val_accuracy: 0.7192
    Epoch 75/100
    169/169 [==============================] - 2s 15ms/step - loss: 0.2804 - accuracy: 0.8819 - val_loss: 1.0404 - val_accuracy: 0.7305
    Epoch 76/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.2724 - accuracy: 0.8845 - val_loss: 1.0342 - val_accuracy: 0.7298
    Epoch 77/100
    169/169 [==============================] - 3s 19ms/step - loss: 0.2575 - accuracy: 0.8912 - val_loss: 1.0535 - val_accuracy: 0.7367
    Epoch 78/100
    169/169 [==============================] - 3s 16ms/step - loss: 0.2659 - accuracy: 0.8887 - val_loss: 1.0776 - val_accuracy: 0.7139
    Epoch 79/100
    169/169 [==============================] - 3s 15ms/step - loss: 0.2606 - accuracy: 0.8894 - val_loss: 1.0805 - val_accuracy: 0.7292
    Epoch 80/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.2579 - accuracy: 0.8914 - val_loss: 1.0501 - val_accuracy: 0.7334
    Epoch 81/100
    169/169 [==============================] - 2s 15ms/step - loss: 0.2625 - accuracy: 0.8885 - val_loss: 1.0818 - val_accuracy: 0.7297
    Epoch 82/100
    169/169 [==============================] - 3s 18ms/step - loss: 0.2575 - accuracy: 0.8912 - val_loss: 1.1078 - val_accuracy: 0.7299
    Epoch 83/100
    169/169 [==============================] - 3s 19ms/step - loss: 0.2488 - accuracy: 0.8959 - val_loss: 1.1052 - val_accuracy: 0.7318
    Epoch 84/100
    169/169 [==============================] - 3s 16ms/step - loss: 0.2465 - accuracy: 0.8960 - val_loss: 1.1513 - val_accuracy: 0.7313
    Epoch 85/100
    169/169 [==============================] - 3s 18ms/step - loss: 0.2509 - accuracy: 0.8942 - val_loss: 1.1090 - val_accuracy: 0.7266
    Epoch 86/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.2417 - accuracy: 0.8981 - val_loss: 1.1801 - val_accuracy: 0.7225
    Epoch 87/100
    169/169 [==============================] - 3s 16ms/step - loss: 0.2525 - accuracy: 0.8932 - val_loss: 1.1006 - val_accuracy: 0.7340
    Epoch 88/100
    169/169 [==============================] - 3s 21ms/step - loss: 0.2305 - accuracy: 0.9032 - val_loss: 1.1466 - val_accuracy: 0.7396
    Epoch 89/100
    169/169 [==============================] - 2s 15ms/step - loss: 0.2412 - accuracy: 0.8988 - val_loss: 1.1200 - val_accuracy: 0.7358
    Epoch 90/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.2354 - accuracy: 0.9011 - val_loss: 1.1577 - val_accuracy: 0.7400
    Epoch 91/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.2283 - accuracy: 0.9032 - val_loss: 1.1570 - val_accuracy: 0.7304
    Epoch 92/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.2382 - accuracy: 0.8992 - val_loss: 1.1771 - val_accuracy: 0.7354
    Epoch 93/100
    169/169 [==============================] - 3s 19ms/step - loss: 0.2416 - accuracy: 0.8984 - val_loss: 1.1889 - val_accuracy: 0.7262
    Epoch 94/100
    169/169 [==============================] - 3s 19ms/step - loss: 0.2280 - accuracy: 0.9044 - val_loss: 1.1669 - val_accuracy: 0.7327
    Epoch 95/100
    169/169 [==============================] - 2s 15ms/step - loss: 0.2291 - accuracy: 0.9039 - val_loss: 1.0788 - val_accuracy: 0.7391
    Epoch 96/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.2210 - accuracy: 0.9068 - val_loss: 1.1809 - val_accuracy: 0.7353
    Epoch 97/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.2161 - accuracy: 0.9096 - val_loss: 1.1822 - val_accuracy: 0.7377
    Epoch 98/100
    169/169 [==============================] - 3s 18ms/step - loss: 0.2176 - accuracy: 0.9083 - val_loss: 1.1763 - val_accuracy: 0.7350
    Epoch 99/100
    169/169 [==============================] - 3s 20ms/step - loss: 0.2323 - accuracy: 0.9029 - val_loss: 1.1882 - val_accuracy: 0.7418
    Epoch 100/100
    169/169 [==============================] - 2s 14ms/step - loss: 0.2154 - accuracy: 0.9094 - val_loss: 1.2146 - val_accuracy: 0.7290
    




    <keras.src.callbacks.History at 0x7d5e455e9150>




```python
test_loss, test_acc = best_model.evaluate(test_X, test_y)
test_loss, test_acc
```

    1686/1686 [==============================] - 7s 4ms/step - loss: 1.2291 - accuracy: 0.7297
    




    (1.2290809154510498, 0.7296951413154602)



### 저장한 모델 로드


```python
from keras.models import save_model, load_model
best_model.save('melon_model.h5')
```

    /usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3000: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
      saving_api.save_model(
    


```python
# 모델 로드
loaded_model = load_model('melon_model.h5')
```


```python
from PIL import Image
```


```python
# 이미지 파일 경로
image_path = 'newjeans.jpg'

# 이미지 불러오기
image = Image.open(image_path)

# resize
new_size = (48, 48)
image = image.resize(new_size)
```


```python
# 로드된 모델을 사용하여 예측 또는 추가 훈련 등의 작업 수행
prediction = loaded_model.predict(np.expand_dims(image, axis=0))
print(prediction)
np.argmax(prediction)
```

    1/1 [==============================] - 0s 18ms/step
    [[0.05103343 0.18057665 0.76838994]]
    




    2



### 모델 예측


```python
# 이미지 파일 경로
image_path = 'pussy.jpg'

# 이미지 불러오기
image = Image.open(image_path)

# resize
new_size = (48, 48)
image = image.resize(new_size)
```


```python
prediction = loaded_model.predict(np.expand_dims(image, axis=0))
print(prediction)
```

    1/1 [==============================] - 0s 23ms/step
    [[0.05681313 0.2808108  0.66237605]]
    


```python

```
