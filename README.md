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
![img1 daumcdn](https://github.com/sjh9824/NLP/assets/73771922/55420b20-232b-4424-8b6c-acb702531c77)  
좋은 성능을 보이는 Neural Sequence Transduction Model은 대부분 위와 같은 형식의 Encoder-Decoder 형식을 띄고 있다.
Transformer 또한 이 형식을 따라가고 있으며 그 내부는 self-attetion과 fully connectied layer로 이루어져 있다.

### 2-1 Attention  
<img width="586" alt="img1 daumcdn" src="https://github.com/sjh9824/NLP/assets/73771922/26709cd9-4633-4fb9-b882-d5da466128e5">  

#### Scaled Dot-Production Attention  
우선 input으로 3가지의 값을 가져간다.  
Query(Q) , Key(K), Values(V)  
Query는 물어보는 주체, Key는 물어봄을 당하는 주체, Values는 데이터 값을 의미한다. 이 3가지 값을 통해 수식으로 나타내자면 다음과 같다.  
$$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$  
여기서 Query q는 어떤 단어를 나타내는 벡터이며, Key k는 그 단어 벡터들을 stack한 matrix이다.  
****Q와 K는 $d_k$ dimentions, V는 $d_v$ dimentions를 가지는데 논문에서는 $d_k = d_v$로 두고 진행한다.**  
![img1 daumcdn](https://github.com/sjh9824/NLP/assets/73771922/6c5bf01a-de2b-4cd1-b69c-d5520be475ab)  
AI is awesome이란는 문장이 있다고 할 때, Query는 awesome을 vector로 가지고 있는 상황에서 $QK^T$는 awesome vector와 key matrix간 dot product를 해줌으로써 Q와 K간 relation Vector를 얻어낸다.  
이 과정을 Q가 모든 단어를 vector로 한번씩 가지는것을 반복하며 그렇게 얻어낸 relation vector들을 stack하여 matrix로 얻어낸다.  
실제 과정에서는 Query 또한 단어들을 하나의 matrix로 stack하여 Query와 Key, 즉 matrix끼리 dot product를 진행하게 된다.  

attention functions에서는 주로 additive attention과 dot product attention 두 연산이 주로 선택되어 사용되어지는데, 이 논문에서는 dot-product attetion을 사용하였다.  
dot-product attetion은 additive attention에 비해 빠르고 공간효율적이라는 장점을 가지고 있다.  
이 논문에서 기존 방식과 다른 점은 $\sqrt{1}{d_k}$로 scailing하는 점인데, 그 이유는 softmax가 0이랑 가까울 때 gradient가 높고, large positive and large negative value에서 gradient가 낮아 scailing으로 이를 해결하였으며, scailing을 진행하지 않는다면 additive attention보다 성능이 떨어지는 현상이 생긴다고 한다.  

