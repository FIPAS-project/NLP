# Attention is All you need. 논문 리뷰
  
#### 0. 개요  
요새 자연어 처리와 비전 처리 분야에서 Transformer가 많이 사용되어지는 추세이다. 이런 Transformer를 처음 제안한 논문이 이 Attetion is All you need라는 논문인데, 이 논문을 읽고 리뷰 그리고 구현해보임으로써 Transformer 모델에 대해 이해하고자 한다.
  
## 1. Sequence Model
![img1 daumcdn](https://github.com/sjh9824/NLP/assets/73771922/39a4d9f2-a076-4b3d-a585-0d471ef4eb55)

Sequence Model이란 어떤 sequence를 가지는 데이터로부터 또 다른 squence를 가지는 데이터를 생산하는 task를 수행한다.  
대표적인 예시로는 machine translation과 chatbot이 있다.  
  
  
![img1 daumcdn](https://github.com/sjh9824/NLP/assets/73771922/b88b4ec6-e7ae-40a0-98c0-003b01fd4eb4)

이러한 구조를 가진 sequence model에는 대부분 Recurrent Neural network, LSTM, GRU가 주측으로 사용되어졌다.  
이 Sequence Model에는 단점이 존재하는데, 해당 구조들의 모델은 데이터를 한번에 처리하지 못한다.  
어느 한 sequence position t에 따라 순차적으로 입력이 되어진다.  
예시로 I am iron man이라는 문장이 있다고 할 떄, 해당 모델에서 iron이라는 단어를 처리하기 위해서는 이전 단어인 I 와 AM을 처리해야 하며, iron을 처리하는 시점인 t 기준으로 I를 network에 넣고 산출되는 h<sub>t-2</sub>를 다음 단어 am에 대해 h<sub>t-1</sub>을 계산할 때 사용되어 진다. 마찬가지로 이 과정으로 산출된 h<sub>t-1</sub>은 h<sub>t</sub>를 계산할 때 사용되어 진다.  
이러한 예시를 통해 알 수 있는 이 구조의 Sequence Model 단점을 알 수 있다. 긴 Sequence를 가진 데이터에 대해서 그 memory와 computation에 부담이 증가된다는 것이다.

#### Attetion Mechanism  
input 또는 output 데이터에서 sequence distance에 무관하게 서로 간 dependencies를 모델링한다.  
![img1 daumcdn](https://github.com/sjh9824/NLP/assets/73771922/0fc1065e-f8dd-4e7c-8b9c-d9a26a2eefc2)

위 이미지는 Franch를 English로 변환하는 모델에서 그 correlation matrix를 나타낸다.  
attention mechanism은 이미 Transformer에서 사용되어지는 mechanism이였으나 이 논문에서는 이 mechanism만을 이용한 Transformer를 제안하고 그 뛰어난 성능과 효율성을 보여준다.  

#### Sequence to Sequence  
![img1 daumcdn](https://github.com/sjh9824/NLP/assets/73771922/a4ef3550-4604-484b-bb0b-561ccdace83d)
논문에서는 언급되지 않았지만, Transformer 모델의 장점을 설명할 때 이 Sequence to Sequence가 소개되어지곤 한다.  
Sequence to Sequence 모델은 위와 같은 구조를 가지고 있으며, 어느 시점 t에서 hiden state h<sub>t</sub>을 구하기 위해서는 이전 t-1의 hidden state h<sub>t-1</sub>을 필요로 한다.  
즉, h<sub>t</sub>은 이전 sequence들의 정보를 담고 있다.  
위의 그림에서 예시를 들자면, tomorrow의 입력을 받아 출력되는 encoder의 마지막 state는 Are, you, free의 정보를 다 담고 있다는 것이다.  
Model에서 encoder의 최종 output은 일종의 embedded vector로 사용하여 decoder에 넣어주게 되는데, memory와 computation 때문에 embedded vector에 maximum lenghth를 제한한다.  
이러한 이유 때문에 긴 Sequence의 데이터를 처리 시 제한된 vector로 정보를 담아내기 때문에 정보릐 손실이 커지고 성능의 방목현상이 일어난다는 단점이 있다.  

## 2.Model Architecture


