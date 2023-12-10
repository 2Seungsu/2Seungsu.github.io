```python
import pandas as pd
import numpy as np
```


```python
df_hiphop = pd.read_csv('힙합.csv')
df_dance = pd.read_csv('댄스.csv')
df_ballad = pd.read_csv('발라드.csv')
df_trot = pd.read_csv('트로트.csv')
```

### 문서 단어 행렬(Document-Term Matrix, DTM)


```python
from sklearn.feature_extraction.text import CountVectorizer
```


```python
cv = CountVectorizer()
dtm = cv.fit_transform(df_hiphop.data)
dtm
```




    <1164x32032 sparse matrix of type '<class 'numpy.int64'>'
    	with 143639 stored elements in Compressed Sparse Row format>




```python
sample_df = pd.DataFrame(dtm.toarray(), columns= cv.get_feature_names_out())
sample_df = sample_df.iloc[:,~sample_df.columns.isin(['새우','genkidama','nihao','glowin','ya','you','they','yacht','슬펐지만','wat','yall','새우던','새워','glowin','이였다고','odin','tym', '지겨내','the', 'my', 'it', 'me','우리', 'while', 'years', 'in', 'to', 'like','up', 'on','in','don','be','that','all','and','나를','love', 'your','우린','with'])]
sample_df
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
      <th>aaaaaa</th>
      <th>aain</th>
      <th>aaron</th>
      <th>ab</th>
      <th>abandon</th>
      <th>abandoned</th>
      <th>abcd</th>
      <th>abide</th>
      <th>ability</th>
      <th>able</th>
      <th>...</th>
      <th>힙찔</th>
      <th>힙플</th>
      <th>힙하</th>
      <th>힙하대</th>
      <th>힙할</th>
      <th>힙합</th>
      <th>힙합씬</th>
      <th>힙합트레인</th>
      <th>힙해</th>
      <th>힛뎀</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>1159</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1160</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1161</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1162</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1163</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1164 rows × 31993 columns</p>
</div>



### LDA 토픽 모델링


```python
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=4)
lda.fit(sample_df)
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LatentDirichletAllocation(n_components=4)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LatentDirichletAllocation</label><div class="sk-toggleable__content"><pre>LatentDirichletAllocation(n_components=4)</pre></div></div></div></div></div>




```python
for n, i in enumerate(lda.components_):
    idx = np.argsort(i)[::-1][:3]
    topic = cv.get_feature_names_out()[idx]
    print(f'Topic {n+1} : {topic}')
```

    Topic 1 : ['xxk' 'waste' 'nignt']
    Topic 2 : ['october' '같다고' '슬퍼하는']
    Topic 3 : ['다를' '이었던' '증기기관차']
    Topic 4 : ['사라졌던' '이었던' 'glow']
    


```python
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
```


```python
from tqdm import tqdm
from konlpy.tag import Okt
# Okt 객체 생성
okt = Okt()
```


```python
box = []
for i in tqdm(df_hiphop['data']):
    words_with_pos = okt.pos(i)
    words = [
        word for word, pos in words_with_pos if 
        ('*' in word and pos == 'Punctuation') or
        (pos == 'Verb') or
        (pos == 'Noun') or
        (pos == 'Adjective') or
        (pos == 'Alpha') and 
        (word not in ['새우','genkidama','glowin','ya','wasting','you','yacht','나','너','내','해','안','속','날','그','난','슬펐지만','wat','yall','새우던','새워','glowin','이였다고','odin','tym', '지겨내','the', 'my', 'it', 'me','우리', 'while', 'years', 'in', 'to', 'like','up', 'on','in','don','be','that','all','and','나를','love', 'your','우린','with']) 
    ]
    words = [i for i in words if len(i) != 1]
    box.append(words)
```

    100%|██████████████████████████████████████████████████████████████████████████████| 1164/1164 [00:20<00:00, 56.85it/s]
    


```python
dic = corpora.Dictionary(box)

corpus = []
for i in box:
    a = dic.doc2bow(i)
    corpus.append(a)
```

### 토픽 모델링 분류 정확성 시각화


```python
lda = LdaModel(corpus, num_topics=4, id2word=dic)

pyLDAvis.enable_notebook()
pyLDAvis_display = gensimvis.prepare(lda, corpus, dic)
pyLDAvis_display
```

    C:\Users\LG\anaconda3\envs\my_python2\lib\site-packages\pyLDAvis\_prepare.py:243: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only.
      default_term_info = default_term_info.sort_values(
    





<link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v1.0.0.css">


<div id="ldavis_el1155219480554976322428628236" style="background-color:white;"></div>
<script type="text/javascript">

var ldavis_el1155219480554976322428628236_data = {"mdsDat": {"x": [0.02433503834424125, -0.01773771970141514, -0.007944891279486917, 0.0013475726366608237], "y": [0.005925329714520836, 0.01740405783642528, -0.02108732622763181, -0.002242061323314312], "topics": [1, 2, 3, 4], "cluster": [1, 1, 1, 1], "Freq": [26.53284668453472, 26.46440707020209, 25.691078461466425, 21.31166778379676]}, "tinfo": {"Term": ["YEA", "diving", "go", "Heart", "\uace0\uc591\uc774", "\ud558\ub098", "FUCK", "oh", "get", "mine", "Jewelry", "Jelly", "bye", "Oh", "**", "dumb", "\uc0dd\uac01", "uh", "spend", "\ubc14\ubcf4", "we", "CAKE", "Dumb", "yeah", "Exit", "dummy", "do", "\ub9dd\ud574", "\ubc84\ub838\uc73c\uba74", "\uc0bc\ucf1c", "Mago", "gums", "\ubbf8\uc6cc\ud55c\ub2e4\uace0", "\uac10\uae34\ub2e4", "Swerve", "\ub8e8\uc774", "tweeted", "Exit", "myo", "vamos", "swerve", "femenino", "\uc790\ub77c\uc9c0", "Nettrix", "rental", "souron", "ceo", "\uba4b\uc9c4\uc9c0", "Spitfire", "Mm", "\uccd0\uc9c0\ub124", "\ub9f4\ub9e4", "na", "\ub0b4\ub51b", "\ub69c\ub8e8", "buzzkill", "helpful", "BC", "balling", "\ube60\uc838\ub4e4\uc5b4", "funk", "Drownin", "few", "\ub69c\ub69c", "bye", "Yeh", "\ucfe0\ucfe0", "Work", "zone", "La", "die", "runnin", "\ub450\uc138\uc694", "oh", "Jelly", "green", "we", "yeah", "do", "need", "more", "away", "\ud560\uae4c", "got", "just", "know", "no", "go", "\uba40\ub9ac", "money", "\uc5c6\uc5b4", "Yeah", "time", "want", "ain", "so", "baby", "\ub2e4\uc2dc", "\ub098\ub97c", "for", "this", "\uadf8\ub0e5", "\ub9e4\uc77c", "\uc9c0\uae08", "\uac19\uc740", "\uc6b0\ub9ac", "\uc0ac\ub791", "\uc774\uc81c", "\uadf8\ub798", "\uc2dc\uac04", "get", "YEA", "\ubb3c\uacb0", "\ubb34\uad81\ud654", "\ud53c\uc5c8\uc2b5\ub2c8\ub2e4", "\uac14\ub2c8", "cat", "\uace0\uc591\uc774", "\ubcf4\ub0b4\ub9ac", "\ub3d9\uae00\ub3d9\uae00", "\uc553\uc774", "\ube60\uc84c\uc2b5\ub2c8\ub2e4", "\uba48\ucdc4\uc2b5\ub2c8\ub2e4", "MODEL", "\uc774\ub807\ub2e4\ub294\ub370", "\ub098\uc600\uc5b4", "\uc5c6\uc5c8\ub2e4\uc9c0\ub9cc", "\ub4e4\uc5c8\ub294\ub370", "NUMB", "Scottie", "ALPHA", "ckin", "PAPAGO", "\uac15\uc9c4", "treasure", "\ub308\uaebc\uc57c", "pippen", "\ubee3\ubee3\ud558\uac8c", "\uccad\uc8fc", "\uc0c8\uc6e0\ub358", "\uad7d\ud798", "ROLE", "buzz", "SKAS", "\ub180\ub7ec\uc640", "GO", "\ubc14\ubcf4", "NEED", "\uc78a\uc5b4", "Love", "skas", "fxxkboys", "fu", "Need", "\uac14\ub358", "MORE", "\uc2dc\uac04", "\uc0ac\ub78c", "\uc6b0\ub9b0", "\ub9e4\uc77c", "\uc138\uc0c1", "\ub3cc\ub824", "\uba38\ub9ac", "\ubaa8\ub4e0", "for", "\uac70\ub9ac", "go", "\ub9c8\uc74c", "is", "\uc5c6\uc5b4", "can", "\uc6b0\ub9ac", "\uacc4\uc18d", "\ud558\ub098", "\uc774\uc81c", "of", "\uc0dd\uac01", "just", "\uc624\ub298", "\ubab0\ub77c", "\ub2e4\uc2dc", "no", "\uc9c0\uae08", "\uc788\uc5b4", "yeah", "we", "\uc0ac\ub791", "You", "wanna", "know", "got", "Yeah", "\ub300\ub2e8\ud558\ub2e4", "\uc798\uc0dd\uacbc\ub2e4", "concrete", "runaway", "\ud560\uc218\uc788\ub2e4", "Thanks", "\ube48\uc9d1", "\ub514\ub2e4", "Thief", "brrrr", "clap", "\ubc84\ub7ed", "MISTA", "ninja", "\uba67\uc5b4", "ROCK", "\uac00\ubcfc\ub824\uace0", "\ub4e4\ub9ac\uc9c0\uac00", "Olololo", "\ubc84\ud168\uc57c\ub9cc", "\uc88b\uae30\uc5d0", "\ub2c8\uc560\ubbf8", "\uac11\uac11\ud574", "\ub098\uc544\uac00\uae30\uc5d0", "\ub2ec\ub824\uc624\ub290\ub77c", "\uc9c0\ubb38", "\uc774\ub904\ub0b4\uc57c\ub9cc", "\ub2f5\uc9c0", "\uc21c\uae08", "\uc544\uc774\ucf58", "cut", "smooth", "tofu", "Welcome", "Eh", "Switchin", "wish", "\ub9c8\uc774\ud06c\ub85c\ud3f0", "\ubd04\ube44", "\uc790\uc720", "well", "moves", "FRAGILE", "Oh", "\ucd5c\uace0", "**", "\uc77c\ub85c", "make", "\uadf8\ub0e5", "\uc0ac\ub791", "\uae30\ubd84", "sh", "so", "out", "\uc6b0\ub9ac", "life", "\uc774\uc81c", "\uc624\ub298", "yeah", "\uc788\uc5b4", "\uc704\ud574", "\uc5c6\ub294", "day", "\uc804\ubd80", "\ub098\ub97c", "know", "\uc6b0\ub9b0", "\ud558\ub294", "\uac19\uc740", "\uc5c6\uc5b4", "is", "\uc2dc\uac04", "say", "\uc54a\uc544", "just", "no", "do", "\uc0dd\uac01", "\ud558\ub098", "wanna", "\ub2e4\uc2dc", "can", "\ub2ec\ub77c\ubd99\uc740", "Dumb", "\ubc84\ub838\uc73c\uba74", "\ub9dd\ud574", "kiss", "diving", "\uba48\ucd94\uae34", "\uc815\uac70", "dummy", "\uae09\ubc1c\uc9c4", "\uc774\ub974\ub2c8", "entertain", "Findin", "consideration", "panther", "skateboard", "\uc54a\ub2e8", "Stagger", "\ub3c4\uc7a0", "\uc601\uc6d0\ud558\uc9c4", "\ubc1b\uac8c\uc9c0\ub9cc", "\ub3c4\uc548", "\uc54c\uace0\uc2ed\uc9c0\ub9cc", "\ub4a4\ub4a4\ub4a4", "\ub2ec\ub9ac\ub294\uac70\ubc16\uc5d0", "\ub9e4\ub2ec\ub9ac\uace0", "favor", "pink", "Bump", "mine", "CAKE", "\uc804\uad6d\uc2dc\ub300", "balcony", "\uc0bc\ucf1c", "spend", "Shades", "\ubc84\ud2bc", "pole", "YA", "\uc77c\uc5b4\uc11c", "dumb", "FUCK", "Take", "Heart", "Get", "\ud558\ub7ec", "\ub354\uc6b1", "\uc5b4\uc11c", "get", "uh", "Jewelry", "\uc0dd\uac01", "go", "\ud558\ub098", "can", "back", "\ub098\ub97c", "\ub2e4\uc2dc", "\ud558\uc9c0", "\uc9c0\uae08", "yeah", "You", "\uc6b0\ub9ac", "\uc774\uc81c", "\uc804\ubd80", "Like", "no", "\ub9c8\uc74c", "we", "\uc54a\uc544", "\ub0b4\uac8c", "\ud558\ub294", "\uacc4\uc18d", "know", "\uc5c6\ub294", "\uc0ac\ub791", "of", "\uc6b0\ub9b0", "\uc2dc\uac04", "\ub9e4\uc77c", "\uc788\uc5b4", "\uc5c6\uc5b4", "so"], "Freq": [64.0, 68.0, 714.0, 107.0, 65.0, 462.0, 61.0, 407.0, 518.0, 65.0, 57.0, 57.0, 87.0, 295.0, 369.0, 84.0, 541.0, 296.0, 73.0, 147.0, 613.0, 41.0, 29.0, 1091.0, 37.0, 33.0, 443.0, 26.0, 24.0, 50.0, 18.01707403450905, 16.380110839834487, 5.502196269189958, 3.9287879870703843, 20.377176695846952, 3.9153485936983334, 3.9082131313100623, 32.12245040402636, 3.8315431847515797, 3.747312691981185, 19.485759056310197, 3.702885220329818, 3.6335220121052485, 12.074092463407833, 2.8284889044013783, 2.8259976380884035, 7.0371034048654115, 6.3567159228826045, 4.962147682377698, 3.5126506692692447, 4.959654863044491, 4.944590566789171, 20.235221772194464, 4.172982226109421, 13.321307925652121, 13.125933543202049, 2.748284061933676, 2.0722437054416893, 5.456371079329334, 9.577676335342044, 31.883178795404305, 24.418823994373835, 17.192949205366904, 12.697574178031111, 59.9875097137062, 22.335757069071768, 7.469851605213416, 44.12051320893157, 29.473443382315743, 52.47227974473906, 71.43009179160876, 31.22176256486174, 10.439263562252798, 197.98005577009542, 34.57356005098919, 27.508626079837175, 249.89873971067402, 376.46087936778355, 178.45309286841646, 132.8910503057493, 117.03527104425405, 92.79080418766372, 37.313719598506516, 169.86903355319174, 187.7709586807015, 195.53020983788178, 194.85288713039782, 213.55576602518073, 93.6319270451107, 95.75256108611705, 187.99043327483972, 139.7349176151648, 134.40986504577552, 108.45129692328207, 74.62859641894629, 147.75697884305112, 113.06128573121704, 166.14817456413104, 160.65494853980402, 134.82267379749834, 122.40531327861883, 140.17575527604387, 134.36840384276826, 139.2489695974521, 120.52597065997693, 170.8136958258199, 145.17119128152527, 150.53584497940332, 111.66417808077195, 140.58048477279712, 120.12398157835396, 63.170893166784744, 11.960940716126748, 7.576731833783974, 6.759933924282218, 3.7554024494612506, 15.665258000846285, 53.849186958209614, 2.873779175571499, 2.8736636633603387, 10.001177359452761, 3.552859532263484, 3.5507639371682984, 10.569442558996215, 2.8285096744725067, 2.807655737178397, 2.80143467435781, 2.79597130083035, 2.7790423445503176, 10.45237336715073, 2.7622775837888307, 10.39698061485644, 8.9710093651907, 10.964639485935184, 7.473009834521513, 2.708024242175123, 7.455388247550041, 4.031034501187944, 4.015359361360849, 2.6648888402417037, 3.967228388226393, 9.79289423773872, 11.093509511490918, 28.565654662008132, 9.059330621449512, 37.20280275665936, 87.77639976376243, 9.879926714078382, 23.640295412033552, 66.12206463868253, 12.87249758542577, 16.260051177159628, 14.852383080768757, 12.730097141021034, 20.51148828814748, 10.1927647621867, 277.52719905720073, 178.70624571585427, 202.18542239436886, 173.08067888150893, 109.56966819661017, 32.9728323287754, 100.30538704712491, 101.4874531564871, 157.99763429034758, 82.3791452489406, 217.62140857196064, 111.11466039617203, 147.85174978287105, 189.15754273933624, 161.41079087601102, 215.9782424998762, 129.79945539490205, 140.89388192287393, 188.9477265360151, 128.61975440912775, 153.35661033973366, 149.20277352339232, 126.0416139117408, 88.87324353066866, 155.2506919483852, 153.3769900573563, 135.64541775783894, 121.33228462541186, 200.49651171636287, 142.7377860891787, 140.4758357069013, 116.44049384360616, 114.83413589250934, 120.49150443354095, 113.6018002850919, 111.83713484018986, 8.753077917672403, 8.635456321727752, 10.801998119332499, 10.670150190370164, 9.885628418153784, 5.274366994524725, 4.457500421977037, 9.599887587194042, 3.611209331337748, 3.609000967248871, 22.148644128690535, 7.816975581109654, 5.6878599694547285, 8.533706590796973, 2.8028432419609617, 4.880274970722607, 2.786616491978723, 2.728154778796711, 11.441722528026473, 2.7226529876799574, 2.7220822071951183, 2.709812252753057, 4.738223454455459, 2.6859071411337823, 2.6404416935790938, 2.6312387443993766, 2.636818300866342, 5.250016987766533, 4.5638868582088215, 5.152904391779576, 24.231311914485474, 11.524822323476714, 18.330724614705975, 23.177470259845713, 15.524498794832226, 17.23145596554381, 24.961491603688362, 5.1474380461953775, 10.106842912722072, 33.741441961108414, 11.917506407761469, 16.659127047985557, 23.077069607366113, 139.59577759174877, 44.89271821002312, 163.60270596623414, 19.065434457790793, 119.80201606458995, 189.35192189971056, 213.4048942358062, 81.04173073082418, 19.045737912403542, 173.55939301019714, 123.70661000227757, 249.93778745953384, 105.70498834966635, 204.6979659718775, 140.8334238814341, 281.44912540513155, 138.57555119125564, 99.80548374979035, 126.22356074936373, 107.15720170022678, 137.24416714992537, 166.10850662288556, 163.2910258797293, 156.90357764905926, 129.53933092239188, 122.48187362001295, 168.95499684917044, 130.38160957874595, 175.6052215021328, 87.87569815026622, 115.35116393559177, 142.3089066870633, 152.54668324882923, 122.97929688996203, 135.49218684894112, 124.20308462088262, 117.00474571963227, 129.26435951832937, 117.91246757015321, 12.679732130383583, 25.298342387746832, 20.809182171241932, 21.902381812066004, 21.551868384912865, 53.77761679420749, 3.14704618860027, 3.1199163541074486, 25.6238704637185, 3.10070653094001, 3.083628716941343, 2.468195508566693, 2.444501943950286, 2.4323376947134494, 3.0293572866212686, 2.420231539340623, 2.4067981351974015, 2.3893901128244672, 2.3920652818015316, 2.9684348216801593, 2.350113849298493, 3.5083244443949297, 2.3412837601709606, 2.903669786218637, 2.3111260948701653, 2.2641675415359552, 2.825745599548592, 4.500371975854818, 2.8172754785354797, 44.69611880748276, 28.150783899828536, 13.786634795032619, 11.991045410724503, 32.17267556587434, 43.68658573576391, 7.105578833255234, 6.563199549883752, 8.045558224378556, 11.033290722830317, 29.17925770963919, 43.548320913803046, 31.644456216350285, 35.1692041168646, 50.79251883466839, 36.562210864470785, 20.57172011626103, 34.05806166935503, 41.765536415054214, 175.808940304497, 103.9103204904177, 27.417769067101062, 161.97051714632568, 199.9757368243241, 140.1594888264261, 153.1044611452934, 104.36397440526211, 160.55115075342988, 162.5419041572542, 123.6836070934186, 139.43447740357723, 232.67444769555894, 114.41283573370751, 180.2657133700182, 147.86840059335037, 107.43799090433622, 56.444414552389205, 126.63065250281872, 82.78850351697366, 122.23540903657816, 90.9431484746088, 83.60633396553992, 94.48352968341115, 91.97989085892891, 107.85373483067112, 88.69974164856693, 108.25456110809793, 89.28385656177483, 99.69650665630991, 108.76035836453845, 91.00882376607451, 87.9186806954744, 94.15076232789484, 89.70262279465456], "Total": [64.0, 68.0, 714.0, 107.0, 65.0, 462.0, 61.0, 407.0, 518.0, 65.0, 57.0, 57.0, 87.0, 295.0, 369.0, 84.0, 541.0, 296.0, 73.0, 147.0, 613.0, 41.0, 29.0, 1091.0, 37.0, 33.0, 443.0, 26.0, 24.0, 50.0, 19.31886245039228, 18.26010321778562, 6.410695628510747, 4.581505032158535, 23.848955933711924, 4.58390512870261, 4.579963659491638, 37.70430977570781, 4.5780558537698735, 4.554289320613367, 23.868918361099688, 4.5442997542201, 4.571217775827399, 15.4497598143423, 3.6241914397118276, 3.625157294842178, 9.030011922027407, 8.157652269983016, 6.375315777430601, 4.517366662230541, 6.383245661948481, 6.369957282502644, 26.117650544062663, 5.410528917517779, 17.335308521857826, 17.244878305212517, 3.6116410918617547, 2.7274387465827106, 7.239601620278255, 12.76223991206075, 42.825714373790646, 32.936418229164644, 23.578375128376535, 17.299950415110363, 87.2285684561122, 31.315976580736532, 10.00981340280984, 66.70326806440545, 45.27577689808948, 86.44159840864577, 122.87375287555824, 49.52563678369692, 14.557422854159745, 407.34041576619285, 57.159872011477546, 44.013848469572025, 613.3087896682107, 1091.0809641848368, 443.9444199185511, 316.10127897223407, 271.7315838353563, 210.37864523393836, 68.40971407927672, 459.2848463672941, 544.0555152423004, 587.1664749818232, 627.4072129394021, 714.7709042312322, 235.735508734146, 244.0477056762734, 640.2537351912413, 423.0367502425416, 401.9261457342527, 302.7998416314328, 177.24592353345454, 500.3151916719286, 337.42783087029295, 613.2051301880998, 595.972094097096, 454.4990629613609, 399.8304303793538, 507.21649386506385, 478.2084608734047, 515.9349368795961, 400.53607026003345, 816.995439155248, 607.3064823323307, 692.0499380806464, 353.744332043544, 702.4732636966692, 518.2000136902147, 64.46054807138937, 13.24849683659744, 8.759592103718164, 7.888038664967091, 4.410429710540316, 18.508775223031975, 65.02835827647532, 3.5275341590262883, 3.528792041937309, 12.311141172497369, 4.388649953096796, 4.390206245414837, 13.135235956039903, 3.531445532335686, 3.525376066299545, 3.526665578290081, 3.5233474975430727, 3.528263225375623, 13.272198526411707, 3.5306630502102982, 13.307763294742314, 11.549392900213611, 14.188109833640889, 9.710485538518407, 3.5323313851546447, 9.734846077896249, 5.281902341396881, 5.269281615837475, 3.523025974817227, 5.289234032384343, 13.07596704210092, 14.941906504089047, 39.78205875270699, 12.394024851649608, 55.95343302939355, 147.77377593341947, 13.85641676691234, 35.79854074517874, 112.06024123022718, 18.586256085787547, 24.056050122919995, 22.12333992566433, 18.618593613597692, 32.62837713503105, 14.749347784736337, 702.4732636966692, 454.76821096566607, 561.7018466138854, 478.2084608734047, 271.5734445420898, 61.38950349134226, 248.80659509142723, 257.6407642672542, 454.4990629613609, 201.73263758033465, 714.7709042312322, 300.6296603179569, 447.95498017352213, 640.2537351912413, 540.5110980346975, 816.995439155248, 410.989038030231, 462.08464626305795, 692.0499380806464, 409.6368614295443, 541.659297925961, 544.0555152423004, 428.0760181374804, 259.91903783437044, 613.2051301880998, 627.4072129394021, 515.9349368795961, 431.375256473909, 1091.0809641848368, 613.3087896682107, 607.3064823323307, 410.4226921938091, 424.6533296418034, 587.1664749818232, 459.2848463672941, 423.0367502425416, 9.743544945550106, 9.73616259578753, 12.33414417218592, 12.350237230704156, 11.495528462391855, 6.177073197884089, 5.29698124154592, 11.490995181985006, 4.407553792615091, 4.408062189100803, 27.289291756636878, 9.683873805310327, 7.056740098366561, 10.587791513903735, 3.52447413242145, 6.170879322250777, 3.5263429002560462, 3.5273148015984, 14.836240113680287, 3.5327404303261893, 3.533337624195932, 3.5237097787747524, 6.207106726895548, 3.534531592509601, 3.5346869151156506, 3.522869631225464, 3.5319058701687225, 7.0333339611528025, 6.163064971012861, 7.033936925420435, 33.32962705066433, 15.736886606776316, 25.461675668499062, 32.622052098661435, 22.10637233685648, 25.179418089710627, 38.62438260578609, 7.02760763932949, 14.64586231981752, 55.1949303195858, 17.592461099645053, 25.57907371381899, 37.07896868707397, 295.2885398615749, 80.75922658724096, 369.10083952293087, 30.615989718959042, 274.81414340456166, 507.21649386506385, 607.3064823323307, 190.27330166531542, 31.516663241476255, 500.3151916719286, 331.8715054719781, 816.995439155248, 289.27257780388436, 692.0499380806464, 428.0760181374804, 1091.0809641848368, 431.375256473909, 277.2196375127792, 391.0134935265023, 311.99860923377565, 447.610051203895, 595.972094097096, 587.1664749818232, 561.7018466138854, 431.32015425758743, 400.53607026003345, 640.2537351912413, 447.95498017352213, 702.4732636966692, 249.17039559954065, 381.7323901788693, 544.0555152423004, 627.4072129394021, 443.9444199185511, 541.659297925961, 462.08464626305795, 424.6533296418034, 613.2051301880998, 540.5110980346975, 13.398793265886466, 29.499609564908415, 24.749265504747694, 26.359090579190404, 27.32542456548344, 68.30908376608109, 4.028286724087541, 4.032221293414716, 33.14374423671344, 4.0314192217559075, 4.023580736481, 3.2367608622567348, 3.238788662880558, 3.237868505714214, 4.044483096096116, 3.2374754077554524, 3.23706592865155, 3.236154106068479, 3.244837133864701, 4.054983471874852, 3.251267046077968, 4.861430219534391, 3.2528681943949307, 4.075584594608129, 3.2549240157743276, 3.256200731228457, 4.074516065893407, 6.512125314334403, 4.0794239175823845, 65.11670764053385, 41.55652894806085, 20.362835506001108, 18.020363798447736, 50.09501405106863, 73.54596752134357, 10.644588348778079, 9.787034362056882, 12.273994390252913, 17.371399962774266, 52.07691508833195, 84.01558605938168, 61.748780688139675, 70.3574881829522, 107.91324025314199, 75.19313420948394, 38.314693054320486, 70.36442724622074, 89.99056816929311, 518.2000136902147, 296.2387979692162, 57.85370055026937, 541.659297925961, 714.7709042312322, 462.08464626305795, 540.5110980346975, 330.93587607864214, 595.972094097096, 613.2051301880998, 431.39684792929074, 515.9349368795961, 1091.0809641848368, 410.4226921938091, 816.995439155248, 692.0499380806464, 447.610051203895, 154.7593410771442, 627.4072129394021, 300.6296603179569, 613.3087896682107, 381.7323901788693, 329.4669359770108, 431.32015425758743, 410.989038030231, 587.1664749818232, 391.0134935265023, 607.3064823323307, 409.6368614295443, 561.7018466138854, 702.4732636966692, 478.2084608734047, 431.375256473909, 640.2537351912413, 500.3151916719286], "Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4"], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -8.1271, -8.2223, -9.3132, -9.65, -8.004, -9.6535, -9.6553, -7.5488, -9.6751, -9.6973, -8.0487, -9.7093, -9.7282, -8.5273, -9.9786, -9.9795, -9.0672, -9.1689, -9.4165, -9.762, -9.417, -9.4201, -8.0109, -9.5897, -8.429, -8.4438, -10.0074, -10.2897, -9.3216, -8.7589, -7.5563, -7.823, -8.1739, -8.477, -6.9242, -7.9122, -9.0075, -7.2314, -7.6349, -7.0581, -6.7497, -7.5773, -8.6728, -5.7302, -7.4753, -7.7039, -5.4973, -5.0876, -5.834, -6.1288, -6.2559, -6.488, -7.399, -5.8833, -5.7832, -5.7427, -5.7461, -5.6545, -6.479, -6.4566, -5.782, -6.0786, -6.1175, -6.3321, -6.7058, -6.0228, -6.2904, -5.9055, -5.9391, -6.1144, -6.211, -6.0755, -6.1178, -6.0821, -6.2265, -5.8778, -6.0405, -6.0042, -6.3029, -6.0726, -6.2298, -6.8699, -8.5341, -8.9907, -9.1048, -9.6926, -8.2643, -7.0296, -9.9602, -9.9602, -8.7131, -9.748, -9.7486, -8.6578, -9.976, -9.9834, -9.9857, -9.9876, -9.9937, -8.669, -9.9997, -8.6743, -8.8218, -8.6211, -9.0045, -10.0196, -9.0069, -9.6218, -9.6257, -10.0356, -9.6377, -8.7341, -8.6094, -7.6636, -8.812, -7.3994, -6.541, -8.7253, -7.8528, -6.8243, -8.4607, -8.2271, -8.3176, -8.4718, -7.9948, -8.6941, -5.3899, -5.83, -5.7066, -5.862, -6.3192, -7.5201, -6.4076, -6.3959, -5.9532, -6.6045, -5.633, -6.3052, -6.0196, -5.7732, -5.9318, -5.6406, -6.1498, -6.0678, -5.7743, -6.1589, -5.983, -6.0105, -6.1792, -6.5286, -5.9707, -5.9829, -6.1057, -6.2173, -5.715, -6.0548, -6.0708, -6.2584, -6.2723, -6.2242, -6.2831, -6.2987, -8.8167, -8.8303, -8.6064, -8.6187, -8.6951, -9.3233, -9.4915, -8.7244, -9.7021, -9.7027, -7.8884, -8.9298, -9.2478, -8.8421, -9.9555, -9.4009, -9.9613, -9.9825, -8.5489, -9.9845, -9.9847, -9.9893, -9.4305, -9.9981, -10.0152, -10.0187, -10.0166, -9.3279, -9.468, -9.3466, -7.7985, -8.5416, -8.0776, -7.843, -8.2437, -8.1394, -7.7688, -9.3476, -8.6729, -7.4674, -8.5081, -8.1732, -7.8473, -6.0474, -7.1819, -5.8887, -8.0383, -6.2003, -5.7425, -5.6229, -6.5912, -8.0393, -5.8296, -6.1682, -5.4649, -6.3255, -5.6646, -6.0386, -5.3462, -6.0547, -6.3829, -6.1481, -6.3118, -6.0644, -5.8735, -5.8906, -5.9305, -6.1221, -6.1782, -5.8565, -6.1157, -5.8179, -6.5102, -6.2382, -6.0281, -5.9587, -6.1741, -6.0772, -6.1642, -6.2239, -6.1243, -6.2162, -8.2592, -7.5685, -7.7639, -7.7126, -7.7288, -6.8144, -9.6528, -9.6614, -7.5557, -9.6676, -9.6731, -9.8958, -9.9054, -9.9104, -9.6909, -9.9154, -9.9209, -9.9282, -9.9271, -9.7112, -9.9448, -9.5441, -9.9485, -9.7333, -9.9615, -9.982, -9.7605, -9.2951, -9.7635, -6.9994, -7.4617, -8.1755, -8.3151, -7.3281, -7.0222, -8.8384, -8.9178, -8.7141, -8.3983, -7.4258, -7.0254, -7.3447, -7.2391, -6.8715, -7.2002, -7.7753, -7.2712, -7.0672, -5.6298, -6.1557, -7.4881, -5.7118, -5.501, -5.8565, -5.7681, -6.1514, -5.7206, -5.7083, -5.9815, -5.8617, -5.3496, -6.0594, -5.6048, -5.8029, -6.1223, -6.766, -5.958, -6.383, -5.9933, -6.289, -6.3731, -6.2508, -6.2777, -6.1185, -6.314, -6.1148, -6.3074, -6.1971, -6.1101, -6.2883, -6.3228, -6.2543, -6.3027], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.257, 1.2181, 1.174, 1.1731, 1.1695, 1.1691, 1.1682, 1.1666, 1.1488, 1.1318, 1.1239, 1.122, 1.0972, 1.0803, 1.0789, 1.0778, 1.0774, 1.0773, 1.0762, 1.0752, 1.0744, 1.0735, 1.0716, 1.0671, 1.0634, 1.0539, 1.0536, 1.0521, 1.044, 1.0397, 1.0317, 1.0276, 1.011, 1.0175, 0.9524, 0.9888, 1.0341, 0.9135, 0.8975, 0.8276, 0.7843, 0.8654, 0.9943, 0.6053, 0.824, 0.8568, 0.429, 0.2627, 0.4154, 0.4603, 0.4844, 0.5082, 0.7206, 0.3321, 0.263, 0.2272, 0.1574, 0.1187, 0.4034, 0.3912, 0.1013, 0.2191, 0.2314, 0.3, 0.4618, 0.1071, 0.2334, 0.021, 0.0159, 0.1116, 0.1431, 0.0407, 0.0573, 0.0171, 0.1258, -0.2383, -0.1043, -0.1987, 0.1737, -0.282, -0.1351, 1.3092, 1.2271, 1.1843, 1.175, 1.1686, 1.1626, 1.1407, 1.1244, 1.124, 1.1216, 1.1181, 1.1172, 1.112, 1.1074, 1.1017, 1.0991, 1.0981, 1.0907, 1.0905, 1.0839, 1.0825, 1.0767, 1.0716, 1.0675, 1.0636, 1.0626, 1.0591, 1.0576, 1.0502, 1.0418, 1.0403, 1.0316, 0.9982, 1.016, 0.9212, 0.8085, 0.9911, 0.9144, 0.8018, 0.962, 0.9377, 0.9309, 0.9492, 0.8652, 0.9598, 0.4007, 0.3953, 0.3076, 0.3131, 0.4217, 0.7078, 0.4209, 0.3977, 0.2728, 0.4338, 0.1402, 0.3341, 0.2209, 0.1101, 0.1208, -0.0011, 0.1768, 0.1416, 0.0312, 0.171, 0.0675, 0.0356, 0.1067, 0.2562, -0.0443, -0.0793, -0.0066, 0.0609, -0.3648, -0.1285, -0.1346, 0.0696, 0.0216, -0.2544, -0.0676, -0.001, 1.2518, 1.2391, 1.2264, 1.2128, 1.2082, 1.201, 1.1865, 1.1792, 1.1597, 1.159, 1.1503, 1.1449, 1.1434, 1.1433, 1.1299, 1.1244, 1.1236, 1.1021, 1.0992, 1.0986, 1.0982, 1.0964, 1.089, 1.0845, 1.0673, 1.0672, 1.0668, 1.0666, 1.0586, 1.0478, 1.0402, 1.0475, 1.0304, 1.0172, 1.0056, 0.9797, 0.9225, 1.0477, 0.9881, 0.8669, 0.9696, 0.9302, 0.8848, 0.6098, 0.7718, 0.5454, 0.8854, 0.5288, 0.3737, 0.3132, 0.5055, 0.8554, 0.3003, 0.3722, 0.1746, 0.3523, 0.1409, 0.2473, 0.0041, 0.2235, 0.3374, 0.2283, 0.2903, 0.1769, 0.0815, 0.0793, 0.0837, 0.1562, 0.1742, 0.0268, 0.1248, -0.0273, 0.3168, 0.1623, 0.018, -0.0551, 0.0753, -0.0267, 0.0452, 0.07, -0.1978, -0.1635, 1.4908, 1.3923, 1.3725, 1.3607, 1.3086, 1.3067, 1.299, 1.2894, 1.2886, 1.2834, 1.2799, 1.2748, 1.2646, 1.2599, 1.2569, 1.255, 1.2495, 1.2426, 1.241, 1.234, 1.2213, 1.2197, 1.2171, 1.2069, 1.2035, 1.1826, 1.1799, 1.1764, 1.1757, 1.1696, 1.1564, 1.1559, 1.1386, 1.1031, 1.025, 1.1417, 1.1463, 1.1236, 1.092, 0.9667, 0.8888, 0.8774, 0.8525, 0.7923, 0.8249, 0.924, 0.8203, 0.7783, 0.465, 0.4983, 0.7992, 0.3387, 0.2721, 0.3529, 0.2845, 0.3919, 0.2343, 0.2182, 0.2966, 0.2375, 0.0006, 0.2685, 0.0347, 0.0026, 0.1189, 0.5373, -0.0544, 0.2563, -0.067, 0.1114, 0.1746, 0.0275, 0.0489, -0.1486, 0.0624, -0.1786, 0.0225, -0.1829, -0.3195, -0.1132, -0.0447, -0.3711, -0.1728]}, "token.table": {"Topic": [1, 2, 3, 4, 2, 1, 4, 1, 3, 4, 1, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 2, 3, 4, 1, 3, 4, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 4, 1, 2, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 1, 4, 1, 4, 1, 2, 3, 4, 2, 1, 2, 3, 4, 1, 2, 4, 1, 2, 3, 4, 1, 3, 4, 1, 2, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 3, 4, 1, 2, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 3, 3, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 2, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 4, 3, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 3, 4, 1, 2, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 4, 4, 1, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 4, 1, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 4, 2, 3, 4, 1, 2, 1, 2, 3, 4, 1, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 1, 1, 2, 3, 4, 1, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 3, 1, 1, 3, 2, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 4, 1, 2, 3, 4, 1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 1, 4, 1, 2, 3, 3, 1, 2, 3, 4, 4, 3, 4, 1, 3, 4, 1, 3, 2, 1, 2, 3, 4, 3, 4, 4, 1, 2, 3, 4, 2, 1, 2, 3, 4, 1, 4, 3, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 2, 3, 4, 1, 2, 3, 3, 4, 3, 4, 1, 2, 3, 4, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 4, 2, 1, 2, 4, 3, 1, 2, 3, 4, 1, 2, 3, 4, 2, 4, 1, 2, 1, 1, 2, 3, 4, 4, 1, 2, 3, 4, 3, 4, 3, 1, 2, 3, 4, 2, 1, 2, 3, 4, 3, 1, 2, 3, 4, 2, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 3, 1, 2, 3, 4, 1, 2, 3, 4, 4, 1, 2, 3, 4, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 2, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 1, 2, 3, 4, 1, 3, 1, 2, 3, 4, 1, 2, 3, 4, 4, 3, 1, 2, 3, 4, 3, 2, 3, 1, 2, 3, 1, 2, 3, 4, 1, 2, 4, 2, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 3], "Freq": [0.19506864328203974, 0.15713862931053202, 0.44432302080909053, 0.2031965034187914, 0.8496987555414867, 0.733288695302822, 0.7353979533899261, 0.048127214919698576, 0.26469968205834216, 0.6737810088757801, 0.7286766834515237, 0.03036152847714682, 0.21253069934002775, 0.067797507475459, 0.0338987537377295, 0.067797507475459, 0.8474688434432375, 0.13570747630077235, 0.09047165086718155, 0.7237732069374524, 0.045235825433590775, 0.8487093435832369, 0.10608866794790461, 0.026522166986976153, 0.18878626477117358, 0.6202977271052846, 0.18878626477117358, 0.46964490111089985, 0.016194651762444823, 0.5182288563982343, 0.6175148205629486, 0.10723202626098152, 0.6612641619427194, 0.08936002188415126, 0.12510403063781178, 0.11969177897182068, 0.25268264449606587, 0.1329908655242452, 0.4920662024397072, 0.4448017674884193, 0.03706681395736828, 0.04633351744671035, 0.47260187795644554, 0.6123176761657566, 0.017494790747593046, 0.384885396447047, 0.5185493704751496, 0.01728497901583832, 0.46669443342763467, 0.6015622218619112, 0.19666457253177866, 0.04627401706630086, 0.1503905554654778, 0.30369733854431125, 0.20677265603016937, 0.12277126451791306, 0.3618521480527964, 0.1160090310112002, 0.5889689266722472, 0.17847543232492338, 0.1160090310112002, 0.14170849231523663, 0.8502509538914198, 0.07613110288591166, 0.8374421317450282, 0.07613110288591166, 0.07613110288591166, 0.06779960813147755, 0.6779960813147754, 0.06779960813147755, 0.1355992162629551, 0.931731878428199, 0.05176288213489995, 0.8854716251934528, 0.2213679062983632, 0.07216872996977794, 0.7216872996977793, 0.07216872996977794, 0.14433745993955588, 0.8502766965978329, 0.21483899820868776, 0.6982267441782353, 0.10741949910434388, 0.05370974955217194, 0.7767111038749078, 0.129451850645818, 0.064725925322909, 0.2776944883754713, 0.13546072603681525, 0.47411254112885337, 0.11514161713129296, 0.06740252195554008, 0.7414277415109409, 0.13480504391108017, 0.1731692754138626, 0.7792617393623816, 0.16205145940777502, 0.8102572970388751, 0.07647617929750683, 0.7647617929750683, 0.07647617929750683, 0.15295235859501366, 0.17595871655394613, 0.7289718257234911, 0.05027391901541318, 0.05027391901541318, 0.1506909345893218, 0.7534546729466092, 0.0753454672946609, 0.0753454672946609, 0.18788890039412237, 0.18788890039412237, 0.6576111513794283, 0.7842748774422457, 0.15685497548844912, 0.6180175400947605, 0.8386111348266112, 0.08386111348266112, 0.04193055674133056, 0.04193055674133056, 0.07942995318137572, 0.07942995318137572, 0.6751546020416936, 0.15885990636275144, 0.2131969231334006, 0.1563444102978271, 0.12791815388004035, 0.49745948731126804, 0.8094448357375323, 0.9075328829116159, 0.12261644325447356, 0.09196233244085517, 0.705044548713223, 0.06130822162723678, 0.6596378449930779, 0.05996707681755253, 0.10494238443071692, 0.1799012304526576, 0.2878294213888726, 0.05756588427777452, 0.05756588427777452, 0.6332247270555197, 0.9773419849025822, 0.015513364839723527, 0.33094051502554606, 0.2647524120204368, 0.2694801336636589, 0.13474006683182946, 0.7025168109728664, 0.031932582316948475, 0.06386516463389695, 0.15966291158474238, 0.22415914555854022, 0.2826354443998985, 0.21441309575164716, 0.27776241949645203, 0.42314090222697853, 0.1692563608907914, 0.2087495117653094, 0.19746575437258998, 0.44206007647109424, 0.1330933563568886, 0.17587336375731707, 0.2519267102469677, 0.3348864250721427, 0.18670659096942469, 0.2578329113387293, 0.21930615447202265, 0.18432573924231843, 0.22058654040474174, 0.2810212090087806, 0.31426027674100193, 0.1664783260512431, 0.11098555070082874, 0.05549277535041437, 0.6659133042049724, 0.6906457374663972, 0.13812914749327945, 0.13812914749327945, 0.9074282141232579, 0.06692586382643587, 0.7361845020907946, 0.13385172765287173, 0.06692586382643587, 0.7538470130038877, 0.11597646353905963, 0.057988231769529816, 0.057988231769529816, 0.6878480417821844, 0.05732067014851537, 0.2178185465643584, 0.0458565361188123, 0.19981088342624015, 0.297866224366895, 0.218311891150892, 0.2830654181871735, 0.05402842640584984, 0.8644548224935974, 0.05402842640584984, 0.05402842640584984, 0.7751927749867653, 0.11074182499810933, 0.11074182499810933, 0.15028821566056622, 0.751441078302831, 0.14657764062444684, 0.8061770234344576, 0.03664441015611171, 0.08107575086198907, 0.08107575086198907, 0.8918332594818796, 0.6176903096807006, 0.09001000807598906, 0.06000667205065938, 0.7200800646079125, 0.09001000807598906, 0.20512911950849694, 0.2820525393241833, 0.3429502466782683, 0.16987255209297403, 0.5778288555400928, 0.12207651877607593, 0.14649182253129112, 0.15463025711636286, 0.07319670714838006, 0.07319670714838006, 0.07319670714838006, 0.7905244372025046, 0.4009510921044058, 0.141909656194256, 0.27706170971259503, 0.18020273802445205, 0.333271495365274, 0.0833178738413185, 0.07141532043541586, 0.5237123498597163, 0.1508580311352235, 0.0301716062270447, 0.0301716062270447, 0.7844617619031622, 0.6179016878638232, 0.7362837577478538, 0.8802236243956768, 0.7209996408760385, 0.12723523074283033, 0.08482348716188688, 0.08482348716188688, 0.29703031535507696, 0.34763548019334933, 0.24422492595861883, 0.11221145246747352, 0.18080452650640572, 0.6780169743990214, 0.13560339487980427, 0.04520113162660143, 0.7472146225209033, 0.07005137086133469, 0.14010274172266937, 0.02335045695377823, 0.20784792077050615, 0.6651133464656197, 0.12470875246230369, 0.2315708159585986, 0.2142030047617037, 0.21613276156135872, 0.33963719673927795, 0.2993966300715143, 0.3049928287644398, 0.11752017255143553, 0.27980993464627507, 0.37014066835562326, 0.24821197760318267, 0.21990710296422325, 0.16112005563715368, 0.6361634115989009, 0.045440243685635776, 0.09088048737127155, 0.24992134027099674, 0.8762272485084178, 0.054764203031776114, 0.8306473217286213, 0.2433280236281271, 0.3303903440088331, 0.29020773460235344, 0.13617439854418123, 0.34555297158649767, 0.27386911045951146, 0.26100277641107805, 0.11947310187831037, 0.07319190943244604, 0.03659595471622302, 0.07319190943244604, 0.8051110037569064, 0.3338065239608027, 0.20437134120049144, 0.27760440513066753, 0.18393420708044228, 0.2627279798762158, 0.21778766752896836, 0.3664363929852484, 0.15210567255991442, 0.1673858536905144, 0.1892187911284076, 0.43665874875786365, 0.21105172856630075, 0.07678520891445685, 0.12285633426313096, 0.1074992924802396, 0.6910668802301116, 0.39336571402700643, 0.1802926189290446, 0.1884877379712739, 0.24175601174576436, 0.4305719576230453, 0.21712602991247582, 0.2097658255086631, 0.14352398587434842, 0.0781889142029216, 0.23456674260876478, 0.6646057707248335, 0.0390944571014608, 0.873733333049254, 0.7657656635790545, 0.07657656635790545, 0.11486484953685819, 0.42075122388759, 0.18664903916817902, 0.2151209264989182, 0.18032195309468144, 0.0944484030202913, 0.0944484030202913, 0.8500356271826216, 0.0944484030202913, 0.3108029298649998, 0.24386076035561524, 0.24386076035561524, 0.2024203697069486, 0.22458916338471066, 0.3149130660503008, 0.24411865585294637, 0.21726560370912226, 0.48607992808071604, 0.1669365409570136, 0.22340037098659174, 0.12274745658603942, 0.26214965299979914, 0.21092500816075793, 0.3736385858847712, 0.15668714891942018, 0.7417511530449246, 0.15355969852097492, 0.15355969852097492, 0.7677984926048746, 0.1027237608071257, 0.7190663256498799, 0.08147306966297174, 0.16294613932594348, 0.16294613932594348, 0.6517845573037739, 0.8277708421050023, 0.08097010456721279, 0.08097010456721279, 0.8906711502393406, 0.625938443464996, 0.20191562692419227, 0.0807662507696769, 0.0807662507696769, 0.2608656612018472, 0.23678575401398433, 0.3531719720886546, 0.14849276099182068, 0.12691698893859926, 0.15864623617324908, 0.6028556974583466, 0.12691698893859926, 0.2152127885001381, 0.6994415626254489, 0.05380319712503453, 0.05380319712503453, 0.6177653103430378, 0.06354497080568651, 0.06354497080568651, 0.7625396496682381, 0.12708994161137302, 0.29581352408153133, 0.1778878624544344, 0.3477807647985571, 0.1798866024820123, 0.8275502980983356, 0.2039540780484925, 0.10877550829252933, 0.09517856975596316, 0.5982652956089113, 0.7960142857149826, 0.12568646616552356, 0.041895488721841186, 0.30512935167102717, 0.210089061806281, 0.2826198093346399, 0.2000848207678867, 0.3333945836123801, 0.23387381238480398, 0.2488019280689404, 0.1841134267710159, 0.11782413848400287, 0.07854942565600191, 0.7069448309040173, 0.07854942565600191, 0.10298146225884566, 0.7208702358119197, 0.10298146225884566, 0.8733693752591887, 0.17215840851912886, 0.3071846112792299, 0.16878275345012633, 0.35106812717626273, 0.878292905524343, 0.2566799607857588, 0.2708091329391033, 0.27551885699021816, 0.1978084101468233, 0.3566712565571858, 0.25429339587873434, 0.23447832607000177, 0.1552180468350716, 0.40762500751904374, 0.23316150430089302, 0.15978900294746515, 0.19892100366929336, 0.11368506024665032, 0.11368506024665032, 0.6821103614799019, 0.11368506024665032, 0.15534228886550994, 0.10356152591033996, 0.6472595369396248, 0.10356152591033996, 0.34461237281407003, 0.18330445362450534, 0.25754275734243, 0.21354968847254874, 0.640519102858812, 0.15460805931074773, 0.15460805931074773, 0.044173731231642206, 0.8507397280571244, 0.8730755443731196, 0.1611056558230222, 0.805528279115111, 0.9069411060878159, 0.21453717943221076, 0.6436115382966323, 0.06129633698063165, 0.09194450547094747, 0.14096310385601019, 0.775297071208056, 0.07048155192800509, 0.30209513945010036, 0.20472563169345645, 0.3045917934951425, 0.1872490533781614, 0.2032392997571989, 0.4064785995143978, 0.1933251875739209, 0.19828224366555988, 0.27008019613368667, 0.31631013961602944, 0.18978608376961767, 0.22385025265134392, 0.06151162517425942, 0.8304069398525021, 0.04613371888069456, 0.06151162517425942, 0.18906329231743377, 0.7562531692697351, 0.27601626069605806, 0.17941056945243775, 0.3726219519396784, 0.16955284585614996, 0.3166128467783151, 0.2657286392603716, 0.1950561288187834, 0.22049823257775514, 0.7441548087607056, 0.17869033491522054, 0.23650191385838013, 0.4257034449450842, 0.15766794257225342, 0.2701468770008715, 0.18289446952232913, 0.2785365315661159, 0.2701468770008715, 0.8487687608614439, 0.8509730433238538, 0.19121797400755247, 0.2519220927401088, 0.30048538772615385, 0.2549572986767366, 0.7392992553924107, 0.18482481384810268, 0.16136808050161414, 0.7261563622572635, 0.08068404025080707, 0.8513754504047566, 0.27070875931693483, 0.25277022707304153, 0.210370059951112, 0.26581643234132757, 0.9702366281818976, 0.848731463929909, 0.61445366782985, 0.14218008209524782, 0.7109004104762391, 0.14218008209524782, 0.10263205081808564, 0.9236884573627708, 0.8492974392516291, 0.18475244535864857, 0.1421172656604989, 0.18475244535864857, 0.48319870324569625, 0.2057007824532296, 0.8228031298129184, 0.6163637549407414, 0.14660486708887077, 0.5375511793258595, 0.11402600773578837, 0.21176258579503554, 0.8501492761112095, 0.6869347754875806, 0.13738695509751614, 0.13738695509751614, 0.06869347754875807, 0.24536357344243787, 0.7360907203273136, 0.8505053188449616, 0.851462991400077, 0.08702466445793562, 0.08702466445793562, 0.8702466445793562, 0.7514472404871958, 0.05780363388363045, 0.17341090165089135, 0.05780363388363045, 0.7499145448498075, 0.05768573421921596, 0.11537146843843192, 0.05768573421921596, 0.872618408909376, 0.1563385327658325, 0.3692250454682427, 0.19958110565850956, 0.27608719616093824, 0.1422959350211263, 0.1422959350211263, 0.7114796751056315, 0.15175030367541825, 0.8346266702148003, 0.30710637412784225, 0.6142127482556845, 0.28021252437746724, 0.3617669158007599, 0.16729105932983118, 0.19029357998768298, 0.7849346201636674, 0.15698692403273348, 0.15272907048961626, 0.40191860655162176, 0.24115116393097305, 0.2049784893413271, 0.3987519763346718, 0.23755436888023, 0.13150331134441304, 0.2333123265787973, 0.7447334823663871, 0.911118926172917, 0.7355057314808163, 0.12258428858013605, 0.12258428858013605, 0.8511908123833737, 0.24452652195461297, 0.39201870980025255, 0.1630176813030753, 0.20183141494666468, 0.20775700175687284, 0.34241431771040154, 0.25392522436951126, 0.19621494610371323, 0.9132845348591355, 0.11416056685739194, 0.07548026106913622, 0.9057631328296346, 0.9359358715013343, 0.1285749103992618, 0.595504848165002, 0.14210911149392094, 0.1285749103992618, 0.6151447948308699, 0.10326446008121584, 0.10326446008121584, 0.8261156806497267, 0.10326446008121584, 0.12121571847958497, 0.8485100293570949, 0.8491991017078491, 0.10217599765224857, 0.20435199530449713, 0.10217599765224857, 0.7152319835657399, 0.8504524307223422, 0.06827866998632687, 0.06827866998632687, 0.6827866998632687, 0.2048360099589806, 0.7551470955997954, 0.783561511843204, 0.0783561511843204, 0.1567123023686408, 0.0783561511843204, 0.9114420249392299, 0.7573029074487087, 0.21769331631553787, 0.3936071072775887, 0.19790301483230716, 0.19130624767123025, 0.23875918373723698, 0.23052610843595295, 0.35072900783469985, 0.17783442650773512, 0.13973446524765823, 0.11977239878370705, 0.09981033231975588, 0.6387861268464375, 0.8515406986619333, 0.16800228547436236, 0.2824653810722796, 0.24923415976965846, 0.29908099172359015, 0.16201878675652567, 0.4050469668913142, 0.2908973671310347, 0.14360756098873867, 0.16225692974248435, 0.8112846487124217, 0.20071938291004393, 0.3957445989290228, 0.25054334320686333, 0.15516604778152332, 0.14216789411147976, 0.14216789411147976, 0.7108394705573988, 0.14216789411147976, 0.6178434557967533, 0.21742980720370225, 0.24100653328603142, 0.3012581666075393, 0.2383868970546615, 0.6148420041876372, 0.08122723848167404, 0.8122723848167405, 0.08122723848167404, 0.08122723848167404, 0.22224551313394716, 0.17779641050715772, 0.1333473078803683, 0.46671557758128907, 0.18157941138977018, 0.26853293233698405, 0.32223951880438084, 0.22761362836182458, 0.29363358566560355, 0.2951954664404206, 0.2639578509440798, 0.14681679283280177, 0.850661888234541, 0.7398303891514821, 0.2149150994262271, 0.2943402448663545, 0.3293807502075872, 0.16118632456967033, 0.2093034940033564, 0.26438336084634495, 0.3059992602388252, 0.22031946737195413, 0.1833713038704007, 0.359621392056514, 0.2795077156082807, 0.17803039210718516, 0.26332910848220437, 0.18396965113140307, 0.360724806140006, 0.19118414725420319, 0.8495104830388848, 0.8493997604349198, 0.7456045240498349, 0.21819234666617884, 0.27310167893978676, 0.2962213977918322, 0.21385739938142032, 0.09798801304607968, 0.19597602609215936, 0.620590749291838, 0.09798801304607968, 0.09601183156719417, 0.17282129682094952, 0.17282129682094952, 0.5568686230897262, 0.19472605055427095, 0.2804982394888903, 0.3222252503219483, 0.20399871962828384, 0.1117363980971406, 0.6704183885828435, 0.1117363980971406, 0.13967049762142575, 0.8750403494561124, 0.2187600873640281, 0.1268232419077095, 0.14494084789452516, 0.6159986035517319, 0.1268232419077095, 0.10270987056365126, 0.9243888350728613, 0.04910907421047973, 0.1473272226314392, 0.09821814842095947, 0.6875270389467162, 0.2435155325641881, 0.20777013328871094, 0.3060699812962731, 0.23904735765475346, 0.7440067847713357, 0.849055572684679, 0.26941381570451484, 0.26359912903463323, 0.19769934677597492, 0.26941381570451484, 0.8515784897088063, 0.7591167623262928, 0.1897791905815732, 0.7833005754119376, 0.15666011508238753, 0.15666011508238753, 0.2600322079275306, 0.12382486091787172, 0.5572118741304227, 0.06191243045893586, 0.6993137352625415, 0.19980392436072614, 0.09990196218036307, 0.8874195851864786, 0.12677422645521122, 0.12335402282020586, 0.30513889855524606, 0.26834910227553554, 0.3029747928917337, 0.22720941517023038, 0.2550309762114831, 0.30140024461357096, 0.21793556148981283, 0.1826975356445054, 0.15659788769529034, 0.13049823974607527, 0.5480926069335161, 0.23875927813195913, 0.23180512439996032, 0.24339538061995833, 0.28743835425595077, 0.540858860440821, 0.1900314915062344, 0.10232464927258776, 0.1607958774283522, 0.08699034613950507, 0.8699034613950507], "Term": ["**", "**", "**", "**", "ALPHA", "BC", "Bump", "CAKE", "CAKE", "CAKE", "Drownin", "Drownin", "Drownin", "Dumb", "Dumb", "Dumb", "Dumb", "Eh", "Eh", "Eh", "Eh", "Exit", "Exit", "Exit", "FRAGILE", "FRAGILE", "FRAGILE", "FUCK", "FUCK", "FUCK", "Findin", "GO", "GO", "GO", "GO", "Get", "Get", "Get", "Get", "Heart", "Heart", "Heart", "Heart", "Jelly", "Jelly", "Jelly", "Jewelry", "Jewelry", "Jewelry", "La", "La", "La", "La", "Like", "Like", "Like", "Like", "Love", "Love", "Love", "Love", "MISTA", "MISTA", "MODEL", "MODEL", "MODEL", "MODEL", "MORE", "MORE", "MORE", "MORE", "Mago", "Mago", "Mm", "Mm", "NEED", "NEED", "NEED", "NEED", "NUMB", "Need", "Need", "Need", "Need", "Nettrix", "Nettrix", "Nettrix", "Oh", "Oh", "Oh", "Oh", "Olololo", "Olololo", "Olololo", "PAPAGO", "PAPAGO", "ROCK", "ROCK", "ROLE", "ROLE", "ROLE", "ROLE", "SKAS", "SKAS", "SKAS", "SKAS", "Scottie", "Scottie", "Scottie", "Scottie", "Shades", "Shades", "Shades", "Spitfire", "Spitfire", "Stagger", "Swerve", "Swerve", "Swerve", "Swerve", "Switchin", "Switchin", "Switchin", "Switchin", "Take", "Take", "Take", "Take", "Thanks", "Thief", "Welcome", "Welcome", "Welcome", "Welcome", "Work", "Work", "Work", "Work", "YA", "YA", "YA", "YA", "YEA", "YEA", "Yeah", "Yeah", "Yeah", "Yeah", "Yeh", "Yeh", "Yeh", "Yeh", "You", "You", "You", "You", "ain", "ain", "ain", "ain", "away", "away", "away", "away", "baby", "baby", "baby", "baby", "back", "back", "back", "back", "balcony", "balcony", "balcony", "balcony", "balling", "balling", "balling", "brrrr", "buzz", "buzz", "buzz", "buzz", "buzzkill", "buzzkill", "buzzkill", "buzzkill", "bye", "bye", "bye", "bye", "can", "can", "can", "can", "cat", "cat", "cat", "cat", "ceo", "ceo", "ceo", "ckin", "ckin", "clap", "clap", "clap", "concrete", "concrete", "concrete", "consideration", "cut", "cut", "cut", "cut", "day", "day", "day", "day", "die", "die", "die", "die", "diving", "diving", "diving", "diving", "do", "do", "do", "do", "dumb", "dumb", "dumb", "dumb", "dummy", "dummy", "dummy", "dummy", "entertain", "favor", "femenino", "few", "few", "few", "few", "for", "for", "for", "for", "fu", "fu", "fu", "fu", "funk", "funk", "funk", "funk", "fxxkboys", "fxxkboys", "fxxkboys", "get", "get", "get", "get", "go", "go", "go", "go", "got", "got", "got", "got", "green", "green", "green", "green", "gums", "gums", "helpful", "is", "is", "is", "is", "just", "just", "just", "just", "kiss", "kiss", "kiss", "kiss", "know", "know", "know", "know", "life", "life", "life", "life", "make", "make", "make", "make", "mine", "mine", "mine", "mine", "money", "money", "money", "money", "more", "more", "more", "more", "moves", "moves", "moves", "moves", "myo", "na", "na", "na", "need", "need", "need", "need", "ninja", "ninja", "ninja", "ninja", "no", "no", "no", "no", "of", "of", "of", "of", "oh", "oh", "oh", "oh", "out", "out", "out", "out", "panther", "pink", "pink", "pink", "pippen", "pippen", "pole", "pole", "pole", "pole", "rental", "runaway", "runaway", "runaway", "runnin", "runnin", "runnin", "runnin", "say", "say", "say", "say", "sh", "sh", "sh", "sh", "skas", "skas", "skas", "skas", "skateboard", "smooth", "smooth", "smooth", "smooth", "so", "so", "so", "so", "souron", "spend", "spend", "spend", "spend", "swerve", "swerve", "swerve", "this", "this", "this", "this", "time", "time", "time", "time", "tofu", "tofu", "tofu", "tofu", "treasure", "treasure", "treasure", "tweeted", "uh", "uh", "uh", "uh", "vamos", "wanna", "wanna", "wanna", "wanna", "want", "want", "want", "want", "we", "we", "we", "we", "well", "well", "well", "well", "wish", "wish", "wish", "wish", "yeah", "yeah", "yeah", "yeah", "zone", "zone", "zone", "zone", "\uac00\ubcfc\ub824\uace0", "\uac10\uae34\ub2e4", "\uac11\uac11\ud574", "\uac11\uac11\ud574", "\uac14\ub2c8", "\uac14\ub358", "\uac14\ub358", "\uac14\ub358", "\uac14\ub358", "\uac15\uc9c4", "\uac15\uc9c4", "\uac15\uc9c4", "\uac19\uc740", "\uac19\uc740", "\uac19\uc740", "\uac19\uc740", "\uac70\ub9ac", "\uac70\ub9ac", "\uac70\ub9ac", "\uac70\ub9ac", "\uacc4\uc18d", "\uacc4\uc18d", "\uacc4\uc18d", "\uacc4\uc18d", "\uace0\uc591\uc774", "\uace0\uc591\uc774", "\uace0\uc591\uc774", "\uace0\uc591\uc774", "\uad7d\ud798", "\uad7d\ud798", "\uadf8\ub0e5", "\uadf8\ub0e5", "\uadf8\ub0e5", "\uadf8\ub0e5", "\uadf8\ub798", "\uadf8\ub798", "\uadf8\ub798", "\uadf8\ub798", "\uae09\ubc1c\uc9c4", "\uae30\ubd84", "\uae30\ubd84", "\uae30\ubd84", "\uae30\ubd84", "\ub098\ub97c", "\ub098\ub97c", "\ub098\ub97c", "\ub098\ub97c", "\ub098\uc544\uac00\uae30\uc5d0", "\ub098\uc600\uc5b4", "\ub0b4\uac8c", "\ub0b4\uac8c", "\ub0b4\uac8c", "\ub0b4\uac8c", "\ub0b4\ub51b", "\ub0b4\ub51b", "\ub180\ub7ec\uc640", "\ub180\ub7ec\uc640", "\ub180\ub7ec\uc640", "\ub2c8\uc560\ubbf8", "\ub2e4\uc2dc", "\ub2e4\uc2dc", "\ub2e4\uc2dc", "\ub2e4\uc2dc", "\ub2ec\ub77c\ubd99\uc740", "\ub2ec\ub824\uc624\ub290\ub77c", "\ub2ec\ub9ac\ub294\uac70\ubc16\uc5d0", "\ub2f5\uc9c0", "\ub2f5\uc9c0", "\ub2f5\uc9c0", "\ub300\ub2e8\ud558\ub2e4", "\ub300\ub2e8\ud558\ub2e4", "\ub308\uaebc\uc57c", "\ub354\uc6b1", "\ub354\uc6b1", "\ub354\uc6b1", "\ub354\uc6b1", "\ub3c4\uc548", "\ub3c4\uc548", "\ub3c4\uc7a0", "\ub3cc\ub824", "\ub3cc\ub824", "\ub3cc\ub824", "\ub3cc\ub824", "\ub3d9\uae00\ub3d9\uae00", "\ub450\uc138\uc694", "\ub450\uc138\uc694", "\ub450\uc138\uc694", "\ub450\uc138\uc694", "\ub4a4\ub4a4\ub4a4", "\ub4a4\ub4a4\ub4a4", "\ub4e4\ub9ac\uc9c0\uac00", "\ub4e4\uc5c8\ub294\ub370", "\ub514\ub2e4", "\ub514\ub2e4", "\ub514\ub2e4", "\ub69c\ub69c", "\ub69c\ub69c", "\ub69c\ub69c", "\ub69c\ub69c", "\ub69c\ub8e8", "\ub69c\ub8e8", "\ub69c\ub8e8", "\ub69c\ub8e8", "\ub8e8\uc774", "\ub9c8\uc74c", "\ub9c8\uc74c", "\ub9c8\uc74c", "\ub9c8\uc74c", "\ub9c8\uc774\ud06c\ub85c\ud3f0", "\ub9c8\uc774\ud06c\ub85c\ud3f0", "\ub9c8\uc774\ud06c\ub85c\ud3f0", "\ub9dd\ud574", "\ub9dd\ud574", "\ub9e4\ub2ec\ub9ac\uace0", "\ub9e4\ub2ec\ub9ac\uace0", "\ub9e4\uc77c", "\ub9e4\uc77c", "\ub9e4\uc77c", "\ub9e4\uc77c", "\ub9f4\ub9e4", "\ub9f4\ub9e4", "\uba38\ub9ac", "\uba38\ub9ac", "\uba38\ub9ac", "\uba38\ub9ac", "\uba40\ub9ac", "\uba40\ub9ac", "\uba40\ub9ac", "\uba40\ub9ac", "\uba48\ucd94\uae34", "\uba48\ucdc4\uc2b5\ub2c8\ub2e4", "\uba4b\uc9c4\uc9c0", "\uba4b\uc9c4\uc9c0", "\uba4b\uc9c4\uc9c0", "\uba67\uc5b4", "\ubaa8\ub4e0", "\ubaa8\ub4e0", "\ubaa8\ub4e0", "\ubaa8\ub4e0", "\ubab0\ub77c", "\ubab0\ub77c", "\ubab0\ub77c", "\ubab0\ub77c", "\ubb34\uad81\ud654", "\ubb34\uad81\ud654", "\ubb3c\uacb0", "\ubb3c\uacb0", "\ubbf8\uc6cc\ud55c\ub2e4\uace0", "\ubc14\ubcf4", "\ubc14\ubcf4", "\ubc14\ubcf4", "\ubc14\ubcf4", "\ubc1b\uac8c\uc9c0\ub9cc", "\ubc84\ub7ed", "\ubc84\ub7ed", "\ubc84\ub7ed", "\ubc84\ub7ed", "\ubc84\ub838\uc73c\uba74", "\ubc84\ub838\uc73c\uba74", "\ubc84\ud168\uc57c\ub9cc", "\ubc84\ud2bc", "\ubc84\ud2bc", "\ubc84\ud2bc", "\ubc84\ud2bc", "\ubcf4\ub0b4\ub9ac", "\ubd04\ube44", "\ubd04\ube44", "\ubd04\ube44", "\ubd04\ube44", "\ube48\uc9d1", "\ube60\uc838\ub4e4\uc5b4", "\ube60\uc838\ub4e4\uc5b4", "\ube60\uc838\ub4e4\uc5b4", "\ube60\uc838\ub4e4\uc5b4", "\ube60\uc84c\uc2b5\ub2c8\ub2e4", "\ubee3\ubee3\ud558\uac8c", "\uc0ac\ub78c", "\uc0ac\ub78c", "\uc0ac\ub78c", "\uc0ac\ub78c", "\uc0ac\ub791", "\uc0ac\ub791", "\uc0ac\ub791", "\uc0ac\ub791", "\uc0bc\ucf1c", "\uc0bc\ucf1c", "\uc0bc\ucf1c", "\uc0bc\ucf1c", "\uc0c8\uc6e0\ub358", "\uc0dd\uac01", "\uc0dd\uac01", "\uc0dd\uac01", "\uc0dd\uac01", "\uc138\uc0c1", "\uc138\uc0c1", "\uc138\uc0c1", "\uc138\uc0c1", "\uc21c\uae08", "\uc21c\uae08", "\uc2dc\uac04", "\uc2dc\uac04", "\uc2dc\uac04", "\uc2dc\uac04", "\uc544\uc774\ucf58", "\uc544\uc774\ucf58", "\uc544\uc774\ucf58", "\uc544\uc774\ucf58", "\uc54a\ub2e8", "\uc54a\uc544", "\uc54a\uc544", "\uc54a\uc544", "\uc54a\uc544", "\uc54c\uace0\uc2ed\uc9c0\ub9cc", "\uc553\uc774", "\uc553\uc774", "\uc553\uc774", "\uc553\uc774", "\uc5b4\uc11c", "\uc5b4\uc11c", "\uc5b4\uc11c", "\uc5b4\uc11c", "\uc5c6\ub294", "\uc5c6\ub294", "\uc5c6\ub294", "\uc5c6\ub294", "\uc5c6\uc5b4", "\uc5c6\uc5b4", "\uc5c6\uc5b4", "\uc5c6\uc5b4", "\uc5c6\uc5c8\ub2e4\uc9c0\ub9cc", "\uc601\uc6d0\ud558\uc9c4", "\uc624\ub298", "\uc624\ub298", "\uc624\ub298", "\uc624\ub298", "\uc6b0\ub9ac", "\uc6b0\ub9ac", "\uc6b0\ub9ac", "\uc6b0\ub9ac", "\uc6b0\ub9b0", "\uc6b0\ub9b0", "\uc6b0\ub9b0", "\uc6b0\ub9b0", "\uc704\ud574", "\uc704\ud574", "\uc704\ud574", "\uc704\ud574", "\uc774\ub807\ub2e4\ub294\ub370", "\uc774\ub904\ub0b4\uc57c\ub9cc", "\uc774\ub974\ub2c8", "\uc774\uc81c", "\uc774\uc81c", "\uc774\uc81c", "\uc774\uc81c", "\uc77c\ub85c", "\uc77c\ub85c", "\uc77c\ub85c", "\uc77c\ub85c", "\uc77c\uc5b4\uc11c", "\uc77c\uc5b4\uc11c", "\uc77c\uc5b4\uc11c", "\uc77c\uc5b4\uc11c", "\uc788\uc5b4", "\uc788\uc5b4", "\uc788\uc5b4", "\uc788\uc5b4", "\uc78a\uc5b4", "\uc78a\uc5b4", "\uc78a\uc5b4", "\uc78a\uc5b4", "\uc790\ub77c\uc9c0", "\uc790\ub77c\uc9c0", "\uc790\uc720", "\uc790\uc720", "\uc790\uc720", "\uc790\uc720", "\uc798\uc0dd\uacbc\ub2e4", "\uc798\uc0dd\uacbc\ub2e4", "\uc804\uad6d\uc2dc\ub300", "\uc804\uad6d\uc2dc\ub300", "\uc804\uad6d\uc2dc\ub300", "\uc804\uad6d\uc2dc\ub300", "\uc804\ubd80", "\uc804\ubd80", "\uc804\ubd80", "\uc804\ubd80", "\uc815\uac70", "\uc88b\uae30\uc5d0", "\uc9c0\uae08", "\uc9c0\uae08", "\uc9c0\uae08", "\uc9c0\uae08", "\uc9c0\ubb38", "\uccad\uc8fc", "\uccad\uc8fc", "\uccd0\uc9c0\ub124", "\uccd0\uc9c0\ub124", "\uccd0\uc9c0\ub124", "\ucd5c\uace0", "\ucd5c\uace0", "\ucd5c\uace0", "\ucd5c\uace0", "\ucfe0\ucfe0", "\ucfe0\ucfe0", "\ucfe0\ucfe0", "\ud53c\uc5c8\uc2b5\ub2c8\ub2e4", "\ud53c\uc5c8\uc2b5\ub2c8\ub2e4", "\ud558\ub098", "\ud558\ub098", "\ud558\ub098", "\ud558\ub098", "\ud558\ub294", "\ud558\ub294", "\ud558\ub294", "\ud558\ub294", "\ud558\ub7ec", "\ud558\ub7ec", "\ud558\ub7ec", "\ud558\ub7ec", "\ud558\uc9c0", "\ud558\uc9c0", "\ud558\uc9c0", "\ud558\uc9c0", "\ud560\uae4c", "\ud560\uae4c", "\ud560\uae4c", "\ud560\uae4c", "\ud560\uc218\uc788\ub2e4", "\ud560\uc218\uc788\ub2e4"]}, "R": 30, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [3, 2, 4, 1]};

function LDAvis_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

if(typeof(LDAvis) !== "undefined"){
   // already loaded: just create the visualization
   !function(LDAvis){
       new LDAvis("#" + "ldavis_el1155219480554976322428628236", ldavis_el1155219480554976322428628236_data);
   }(LDAvis);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/LDAvis
   require.config({paths: {d3: "https://d3js.org/d3.v5"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v3.0.0.js", function(){
        new LDAvis("#" + "ldavis_el1155219480554976322428628236", ldavis_el1155219480554976322428628236_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("https://d3js.org/d3.v5.js", function(){
         LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v3.0.0.js", function(){
                 new LDAvis("#" + "ldavis_el1155219480554976322428628236", ldavis_el1155219480554976322428628236_data);
            })
         });
}
</script>



### tf_idf 모델


```python
df = pd.concat([df_ballad,df_dance, df_hiphop, df_trot])
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
      <th>data</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>잠깐 날 떠난줄 알았는데 날 기다려도 오지 않는 너 년 잊혀진단 친구 위로 사실 될...</td>
      <td>발라드</td>
    </tr>
    <tr>
      <th>1</th>
      <td>수만 가지 생각 돌다 저편 사라진다 열 들떠 붉어진 얼굴 위로 찬 바람 겨 붙은 모...</td>
      <td>발라드</td>
    </tr>
    <tr>
      <th>2</th>
      <td>힘든 거 였니 아픈 거 였니 너 이별 하는 슬픈 거 였니 너 울 웃던 이 계절 전부...</td>
      <td>발라드</td>
    </tr>
    <tr>
      <th>3</th>
      <td>마음 문 활짝 열고 귀 기울여 기다리면 침묵 저편 들려오는 내 음성 다정한 그대 모...</td>
      <td>발라드</td>
    </tr>
    <tr>
      <th>4</th>
      <td>넌 꿈 뭐 난 꿈 찾고 있어요 여기저기 둘러보고 것 것 만져 보고 경험 하며 꿈 찾...</td>
      <td>발라드</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>이태원 프리덤 이태원 프리덤 나를 사랑 채워줘요 사랑 배터리 됐나 봐요 당신 없인 ...</td>
      <td>트로트</td>
    </tr>
    <tr>
      <th>996</th>
      <td>바람 불면 꽃 바람 꿈 그리던 님 찾아 오려나 설레는 가슴 나 사랑 해 영원히 사랑...</td>
      <td>트로트</td>
    </tr>
    <tr>
      <th>997</th>
      <td>목포 행 완행열차 마지막 기차 떠나가고 늦은 밤 홀로 한잔 술 몸 기댄다 우리 사랑...</td>
      <td>트로트</td>
    </tr>
    <tr>
      <th>998</th>
      <td>나 간직 싶기에 이름 밝힌 적도 없었지요 기억 문 열고 들어와 내 앞 서 있는 그대...</td>
      <td>트로트</td>
    </tr>
    <tr>
      <th>999</th>
      <td>봄 봄 봄 봄 왔네요 우리 처음 만났던 그때 향기 그대로 그대 앉아 있었던 그 벤치...</td>
      <td>트로트</td>
    </tr>
  </tbody>
</table>
<p>4114 rows × 2 columns</p>
</div>




```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.8, min_df=0.005,
                        stop_words=['you', 'the', 'my', 'it', 'me','우리', 'while','하는', 'years', 'in', 'to', 'like','up', 'on','in','don','시간','be','that','all','and','사랑','나를','love', 'your','can','우린','with','지난', '이번', '위한'],
                       max_features=50,
                       ngram_range=(1,3))

```


```python
a = tfidf.fit_transform(df['data'])
df_dtm = pd.DataFrame(a.toarray(), columns= tfidf.get_feature_names_out())
df_dtm
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
      <th>baby</th>
      <th>do</th>
      <th>for</th>
      <th>get</th>
      <th>go</th>
      <th>got</th>
      <th>is</th>
      <th>just</th>
      <th>know</th>
      <th>la</th>
      <th>...</th>
      <th>않아</th>
      <th>없는</th>
      <th>없어</th>
      <th>오늘</th>
      <th>이제</th>
      <th>있어</th>
      <th>지금</th>
      <th>하나</th>
      <th>하루</th>
      <th>하지</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.312140</td>
      <td>0.482149</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.173994</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.313591</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.723191</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>4109</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.096474</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.160065</td>
      <td>0.050137</td>
      <td>0.107554</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4110</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4111</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.419368</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4112</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.541179</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4113</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.300846</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>4114 rows × 50 columns</p>
</div>




```python
from sklearn.model_selection import train_test_split
```


```python
data = df_dtm.to_numpy()
target = df['label'].to_numpy() 
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size = 0.2)
```


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
rf = RandomForestClassifier()
rf.fit(train_input, train_target)
rf.score(test_input, test_target)
```




    0.6257594167679222




```python
rf = RandomForestClassifier(oob_score=True)
rf.fit(train_input, train_target)
rf.score(test_input, test_target)
```




    0.6464155528554071




```python
rf.predict(train_input)
```




    array(['힙합', '힙합', '댄스', ..., '트로트', '트로트', '댄스'], dtype=object)




```python
from sklearn.metrics import classification_report
result = classification_report(test_target, rf.predict(test_input))
print(result)
```

                  precision    recall  f1-score   support
    
              댄스       0.59      0.44      0.50       183
             발라드       0.58      0.63      0.60       193
             트로트       0.70      0.83      0.76       203
              힙합       0.68      0.66      0.67       244
    
        accuracy                           0.65       823
       macro avg       0.64      0.64      0.63       823
    weighted avg       0.64      0.65      0.64       823
    
    


```python
import joblib
joblib.dump(rf, 'tfidf_rf.pkl')
```




    ['tfidf_rf.pkl']




```python
loaded_model = joblib.load('tfidf_rf.pkl')
```

### 힙합 예측


```python
#다이나믹듀오 smoke
text = """Light it up Light it up Light it up

나는 달리거나 넘어지거나
둘 중에 하나야 브레이크 없는 bike
택도 없는 것들을 택도 안 뗀 옷 위로 stack it up
난 절대 빠꾸 없는 type

I’m gonna smoke you up
I’mma smoke you
I’m gonna smoke you up
I’mma smoke you
I’m gonna smoke you up
I’mma smoke you
싹 다 부수고 원상복구해 (light it up)

적자생존
아마 난 진짜 1
내일 없는 애들 빈 수레가 요란해 that’s why I’m shooting
To your 골대 cuz you have no keeper
차린 것 없는 밥상 들이밀지 말고 zip up
저기 빈털터리들 재떨이에 털어 넣고 twerkin‘
Then I’mma smoke another chance
You know that I’mma chop it
Lazy ho
그리고 또 stupid thug
모자란 애들 들이 마시고 뱉어 that’s wassup

내 입김은 태풍 내가 후 하고 불면 넌 힘없이 쓰러지는 가로수
무대 위 조명은 늘 파란불 내가 짓밟고 가는 넌 횡단보도 위에 가로줄
어차피 너무 기운 시소 이쯤 되면 너에게 필요한 건 시도 아닌 기도
난 입으로 널 패지 구타
처맞은 것처럼 네 뺨은 붉게 불타

나는 달리거나 넘어지거나
둘 중에 하나야 브레이크 없는 bike
택도 없는 것들을 택도 안 뗀 옷 위로 stack it up 난 절대 빠꾸 없는 type

I’m gonna smoke you up
I’mma smoke you
I’m gonna smoke you up
I’mma smoke you
I’m gonna smoke you up
I’mma smoke you
싹 다 부수고 원상복구해 (light it up)

끝났어 파티는
잔 돌려 ice water
여긴 아무도 없네
날 상대할 카리스마가
타자 팔자에 난 outsider
죽거나 싹 쓸어 주먹 안에 주사위
타고난 dice roller
지그시 밟아주지 부득이 싸움 나면
속도 조절 어린이 보호구역부터 아우토반
까다롭게 굴어 사우스포
갈기지 턱주가리 카운터
넌 피식 쓰러지는 나무토막

I’mma south side baddie
Collect all these veggies
Lap top 위에서 money dance
넌 계속해라 copy
Nothing's dynamic in ur life
저주 같지
다 끝난 파티 뒤에서
꽁초나 하나 줍길
Man I can't curse to you
Cuz you already die for it
Sorry that I’m so stable in my life
I’m done with it
Better get your money
Or u better get ma number
다 피고 남은 꽁초 더밀
꽂아줄게 주머니에

나는 달리거나 넘어지거나
둘 중에 하나야 브레이크 없는 bike
택도 없는 것들을 택도 안 뗀 옷 위로 stack it up 난 절대 빠꾸 없는 type

I’m gonna smoke you up
I’mma smoke you
I’m gonna smoke you up
I’mma smoke you
I’m gonna smoke you up
I’mma smoke you
싹 다 부수고 원상복구해 (light it up)

하이킥 로우킥에 넌 쓰러지지 픽픽
풀린 두 다리 눈 콧바람은 씩씩
네 목을 조르는 내 두 다리 사이로 보이는 네 흰자위
힘없이 탭탭 질식
팔다리를 꺾는 암바와 니바
네 자존심을 꺾는 짬바와 이빨
넌 피투성이 사람들이 기겁해
난 관대해 더 버티지 마 받아줄게 기권패

다 겪었지 대우차부터 테슬라
다 꺾였지 같이 짬밥 먹던 랩 스타
We stand strong
완력보다 강한 펜촉
우습게 봐도 오래 버티는 게 센 놈
상어 밥도 안돼 넌 그냥 벵에돔
엄마 젖은 사치 이유식을 맥여 더
빈약한 커리어 세치 혀로 채 썰어
태운 다음 재 털어

I’m gonna smoke you up
I’mma smoke you
I’m gonna smoke you up
I’mma smoke you
I’m gonna smoke you up
I’mma smoke you
싹 다 부수고 원상복구해 (light it up)

I’m gonna smoke you up
I’mma smoke you
I’m gonna smoke you up
I’mma smoke you
I’m gonna smoke you up
I’mma smoke you
싹 다 부수고 원상복구해 (light it up)"""
```


```python
t = okt.pos(text)
t = [word for word, pos in t if ('*' in word and pos == 'Punctuation') or (pos == 'Verb') or (pos == 'Noun') or (pos == 'Noun') or (pos == 'Adjective') or (pos == 'Alpha')]
t = " ".join(t)

g = tfidf.transform([t]).toarray()
rf.predict(g)
```




    array(['힙합'], dtype=object)



### 트로트 예측


```python
# 장윤정 어머나
text = """어머나 어머나 이러지 마세요   
여자의 마음은 갈대랍니다
안돼요 왜이래요 묻지말아요
더이상 내게 원하시면 안돼요
오늘 처음 만난 당신이지만
내 사랑인걸요
헤어지면 남이 되어
모른척 하겠지만
좋아해요 사랑해요
거짓말처럼 당신을 사랑해요
소설속에 영화속에
멋진 주인공은 아니지만
괜찮아요 말해봐요
당신 위해서라면 다 줄게요
어머나 어머나 이러지 마세요
여자의 마음은 바람입니다
안돼요 왜이래요 잡지말아요
더이상 내게 바라시면 안돼요
오늘 처음 만난 당신이지만
내 사랑인걸요
헤어지면 남이 되어
모른 척 하겠지만
좋아해요 사랑해요
거짓말처럼 당신을 사랑해요
소설속에 영화속에
멋진 주인공은 아니지만
괜찮아요 말해봐요
당신 위해서라면 다 줄게요
소설속에 영화속에
멋진 주인공은 아니지만
괜찮아요 말해봐요
당신 위해서라면 다 줄게요"""
```


```python
t = okt.pos(text)
t = [word for word, pos in t if ('*' in word and pos == 'Punctuation') or (pos == 'Verb') or (pos == 'Noun') or (pos == 'Noun') or (pos == 'Adjective') or (pos == 'Alpha')]
t = " ".join(t)

g = tfidf.transform([t]).toarray()
rf.predict(g)
```




    array(['트로트'], dtype=object)



### 댄스


```python
# 뉴진스 supershy
text = """I’m super shy, super shy
But wait a minute while I
Make you mine, make you mine
떨리는 지금도
You’re on my mind
All the time
I wanna tell you but I’m
Super shy, super shy

I’m super shy, super shy
But wait a minute while I
Make you mine, make you mine
떨리는 지금도
You’re on my mind
All the time
I wanna tell you but I’m
Super shy, super shy

And I wanna go out with you
Where you wanna go? (Huh?)
Find a lil spot
Just sit and talk
Looking pretty
Follow me
우리 둘이 나란히
보이지? (봐)
내 눈이 (heh)
갑자기
빛나지
When you say
I’m your dream

You don’t even know my name
Do ya?
You don’t even know my name
Do ya-a?
누구보다도

I’m super shy, super shy
But wait a minute while I
Make you mine, make you mine
떨리는 지금도
You’re on my mind
All the time
I wanna tell you but I’m
Super shy, super shy

I’m super shy, super shy
But wait a minute while I
Make you mine, make you mine
떨리는 지금도
You’re on my mind
All the time
I wanna tell you but I’m
Super shy, super shy

나 원래 말도 잘하고 그런데 왜 이런지
I don’t like that
Something odd about you
Yeah you’re special and you know it
You’re the top babe

I’m super shy, super shy
But wait a minute while I
Make you mine, make you mine
떨리는 지금도
You’re on my mind
All the time
I wanna tell you but I’m
Super shy, super shy

I’m super shy, super shy
But wait a minute while I
Make you mine, make you mine
떨리는 지금도
You’re on my mind
All the time
I wanna tell you but I’m
Super shy, super shy

You don’t even know my name
Do ya?
You don’t even know my name
Do ya-a?
누구보다도
You don’t even know my name
Do ya?
You don’t even know my name
Do ya-a?"""
```


```python
t = okt.pos(text)
t = [word for word, pos in t if ('*' in word and pos == 'Punctuation') or (pos == 'Verb') or (pos == 'Noun') or (pos == 'Noun') or (pos == 'Adjective') or (pos == 'Alpha')]
t = " ".join(t)

g = tfidf.transform([t]).toarray()
rf.predict(g)
```




    array(['댄스'], dtype=object)



### 발라드


```python
# 박재정 헤어지자말해요
text = """헤어지자고 말하려 오늘
너에게 가다가 우리 추억 생각해 봤어
처음 본 네 얼굴
마주친 눈동자
가까스로 본 너의 그 미소들
손을 잡고 늘 걷던 거리에
첫눈을 보다가 문득 고백했던 그 순간
가보고 싶었던 식당
난생처음 준비한 선물
고맙다는 너의 그 눈물들이
바뀔까 봐 두려워
그대 먼저 헤어지자 말해요
나는 사실 그대에게 좋은 사람이 아녜요
그대 이제 날 떠난다 말해요
잠시라도 이 행복을 느껴서 고마웠다고
시간이 지나고 나면 나는
어쩔 수 없을 걸 문득 너의 사진 보겠지
새로 사귄 친구 함께
웃음 띤 네 얼굴 보면
말할 수 없을 묘한 감정들이
힘들단 걸 알지만
그대 먼저 헤어지자 말해요
나는 사실 그대에게 좋은 사람이 아녜요
그대 이제 날 떠난다 말해요
잠시라도 이 행복을 느껴서 고마웠다고
한 번은 널 볼 수 있을까
이기적인 거 나도 잘 알아
그땐 그럴 수밖에 없던
어린 내게 한 번만 더 기회를 주길
그댈 정말 사랑했다 말해요
나는 사실 그대에게
좋은 사람이 되고 싶었어
영영 다신 못 본다 해도
그댈 위한 이 노래가
당신을 영원히 사랑할 테니"""
```


```python
t = okt.pos(text)
t = [word for word, pos in t if ('*' in word and pos == 'Punctuation') or (pos == 'Verb') or (pos == 'Noun') or (pos == 'Noun') or (pos == 'Adjective') or (pos == 'Alpha')]
t = " ".join(t)

g = tfidf.transform([t]).toarray()
rf.predict(g)
```




    array(['트로트'], dtype=object)




```python
# 임영웅 사랑은 늘 도망가
text = """눈물이 난다 이 길을 걸으면
그 사람 손길이 자꾸 생각이 난다
붙잡지 못하고 가슴만 떨었지
내 아름답던 사람아
사랑이란 게 참 쓰린 거더라
잡으려 할수록 더 멀어지더라
이별이란 게 참 쉬운 거더라
내 잊지 못할 사람아
사랑아 왜 도망가
수줍은 아이처럼
행여 놓아버릴까 봐
꼭 움켜쥐지만
그리움이 쫓아 사랑은 늘 도망가
잠시 쉬어가면 좋을 텐데
바람이 분다 옷깃을 세워도
차가운 이별의 눈물이 차올라
잊지 못해서 가슴에 사무친
내 소중했던 사람아
사랑아 왜 도망가
수줍은 아이처럼
행여 놓아버릴까 봐
꼭 움켜쥐지만
그리움이 쫓아 사랑은 늘 도망가
잠시 쉬어가면 좋을 텐데
기다림도 애태움도 다 버려야 하는데
무얼 찾아 이 길을 서성일까
무얼 찾아 여기 있나
사랑아 왜 도망가
수줍은 아이처럼
행여 놓아버릴까 봐
꼭 움켜쥐지만
그리움이 쫓아 사랑은 늘 도망가
잠시 쉬어가면 좋을 텐데
잠시 쉬어가면 좋을 텐데"""
```


```python
t = okt.pos(text)
t = [word for word, pos in t if ('*' in word and pos == 'Punctuation') or (pos == 'Verb') or (pos == 'Noun') or (pos == 'Noun') or (pos == 'Adjective') or (pos == 'Alpha')]
t = " ".join(t)

g = tfidf.transform([t]).toarray()
rf.predict(g)
```




    array(['트로트'], dtype=object)


