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
이 논문에서 기존 방식과 다른 점은 $1 \over \sqrt{d_k}$로 scailing하는 점인데, 그 이유는 softmax가 0이랑 가까울 때 gradient가 높고, large positive and large negative value에서 gradient가 낮아 scailing으로 이를 해결하였으며, scailing을 진행하지 않는다면 additive attention보다 성능이 떨어지는 현상이 생긴다고 한다.  

![img1 daumcdn](https://github.com/sjh9824/NLP/assets/73771922/1bbcfa1d-ab69-41d8-ac29-3d4320b8ef57)
softmax를 통해 Query와 Key 간에 correlation 정도를 확률 분포로 나타내며 이를 Value matrix와 dot product를 해줌으로써 세 input인 Q, K, V의 correlation matrix가 완성된다.  
옵션으로는 mask layer가 있으며 이는 원하지 않은 correlation을 masking 할 때 사용한다.  

#### Multi-head Attention  
하나의 attention function을 사용하는 것보다 Query, Key, Value를 중간중간 linear projection을 통해 중간중간 매핑을 해주어 각 다른 값들을 입력으로 하는, 여러개의 attention function을 사용하는 것이 더 효율적이다.  
여러 input으로 구해진 output들은 나중에 concatenate되고 linear function을 통해 매핑한다.  
이런 기법은 CNN이 여러개의 필터를 통해서 convolution output을 구하는 것과 비슷한 효과를 보인다.  
$$MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O$$
$$where \ head_i = Attention(QW_i^Q,KW_i^K,VW_i^V)$$
$W_i^Q \in \mathbb{R}^{d_{model} \times d_k},W_i^K \in \mathbb{R}^{d_{model} \times d_k},W_i^V \in \mathbb{R}^{d_{model} \times d_v},W_i^O \in \mathbb{R}^{d_{model} \times hd_v}$  
또한 header가 8개라고 했을 때, $d_k = d_v=d_{model}/h = 64$로 header의 개수대로 나누어져 dimension이 줄어든다.(이 논문에선 $d_{model}$을 512로 설정하였다.)  
이로 인해 single head attention을 했을 때와 비슷한 computation cost를 가지게 된다.  

#### Different use for Multi-head Attention  
이 논문에선 Transformer를 세가지 다른 방법으로 Multi-head attention layer을 사용한다.  
* self-attention in encoder  
  encoder에서 사용되는 self-attention으로 Queries, Keys, Values를 모두 encoder에서 가져온다.
  encoder의 각 position을 이전 layer의 positions를 참조하여 해당 position과 모든 position의 correlation을 나타낸다.
  간단히 말하자면, 어떤 입력 문장의 해당 position의 한 단어가 모든 단어와 어떤 correlation을 가지고 있는지 나타나게 된다.

* self-attention in decoder  
  self-attention in encoder와 같은 방식으로 동작하나, 이 layer에서는 해당 position의 단어가 masking vector를 사용하여 그 이전의 단어 벡터만 참조한다.
  이는 이후에 나올 단어 벡터를 참조하여 학습하는 것은 일종의 치팅이며, sequence model의 auto-regressive property를 보존하기 위함이다.

* encoder-decoder attention  
  decoder에서 self-attention 다음으로 사용되는 layer로 queries는 이전 decoder layer에서, keys와 values는 encoder의 output에서 가져온다.
  decoder의 모든 postion vector가 encoder의 position vector를 참조하게 되어 decoder sequence vector와 encoder sequence vector의 correlation vector를 나타내고 학습한다.

### 2-2 Position-Wise Feed-Forward Networks  
attention layer 이후 fully connected feed-forward network가 사용된다.  
$$FFN(x) = \max{(0,xW_1+b_1)}W_2 + b_2$$  

### 2-3 Embedding and Softmax  
다른 sequence model 처럼 input과 output token을 $d_{model}$ 차원의 vector로 embedding하는 과정을 거친다.  
또한 보통 다음 token을 예측하기 위해 decoder output에 학습된 linear transformation 과 softmax function을 사용하여 변환한다.  
이 논문에서는 embedding layer와 pre-softmax linear transformation에 같은 weight matrix를 공유한다. 

### 2-4 Positional Encoding  
이 논문에서 제안하는 model은 다른 model과는 달리 recurrence나 convolution을 사용하지 않고 오로지 attention만 이용하여 model을 구축한다.  
이에 따라, sequence에 대한 정보를 따로 받아야 했고 해당 역할을 이 논문에서 담당한 기법이 position encoding이다.  
encoding과 decoding 블록 아래에 위치 시켜 embedding된 input과 output이 이 position encoding을 거쳐 encoder와 decoder에 들어가게 된다.  
이 논문에서 어떤 방식으로 position encoding을 진행했는지 알아보기 전에 position encoding에 대해 더 자세하게 알아볼 필요가 있다고 생각했다.  

#### About P.E  
언어를 이해하는 어순은 상당히 중요한 역할을 한다고 생각한다.  
그렇기에 언어 모델에 어순에 대한 정보는 빼놓을 수 없다고 생각한다.  
이 논문의 저자가 선택한 방식은 attention layer에 들어가기 전 입력 값으로 주어질 단어 vector 안에 positional encoding 정보를 주는 것이었다.  
그럼 어떻게 어순에 대한 정보를 주는지 그 방법에 대해 두 가지 방법과 각 방법의 한계를 다음과 같이 제시한다.  
* 데이터에 0~1 사이 label 부여. (첫번째 단어 : 0 , 마지막 단어 : 1)
  ex) I love you : I 0 / love 0.5 / you 1
  한계점: input의 총 크기를 알 수 없다. 따라서 delta 값이 일정한 의미를 가지지 못한다.(delta = 단어의 label 간 차이)

* 각 time-step 마다 선형적 숫자 할당
  ex) I love you : I 1 / love 2 / you 3
  총 input 크기에 따라 가변적이며 delta가 일정해진다.  
  한계점: 숫자가 매우 커질 수 있음. 학습 데이터보다 큰 값이 입력 값으로 들어게 될 때 발생 모델의 일반화가 어려워짐.

Positional Encoding에 대해 참고한 글의 저자에 따르면 이상적인 모델은 다음과 같은 기준을 충족 시켜야 한다고 말했다.  
1. 각 time-step마다 하나의 유일한 encoding 값을 출력.
2. 서로 다른 길이의 문장에 있어 두 time-step간 거리는 일정.
3. 모델에 대한 일반화가 가능해야 함. 더 긴 길이의 문장이 나왔을 때 적용이 가능해야 함. 즉, 순서를 나타내는 값들이 특정 범위 내에 있어야 한다.
4. 하나의 ket 값처럼 결정되어야 한다. 매번 다른 값이 아닌 일정한 값이 출력되어야 한다.

이 논문의 저자가 사용한 encoding 기술은 위의 기준을 모두 충족시킨다.  

#### How P.E used in this paper  
이 논문에서 Position encoding 방식으로 sin과 cosine 함수를 사용한다. 다음은 이 논문에서 정의한 P.E 수식이다.  
$$PE_{(pos,2i)}=\sin(pos/10000^{2i/d_{model}})$$
$$PE_{(pos,2i+1)}=\cos(pos/10000^{2i/d_{model}})$$  
여기서 pos는 position, i는 dimension을 의미한다.  
위 수식을 이해한 방식대로 잠깐 바꿔보고자 한다.  
먼저 t를 한 input문장의 위치라 한다. $R^d$(d는 encoding dimension.에 속하는 $\vec{p_t}$를 이에 상응하는 encoding 값이라 할 때, $f:N \to R^d$는 출력 벡터 $\vec{p_t}$를 만들어 내는 함수이다.  
 $\vec{p_t}$는 다음과 같이 정의한다.  
 
$${\vec{p_t}}^{(i)}={f(t)}^{(i)}:=
\begin{cases}
\sin(\omega_k.t), & \mbox{if }i\mbox{ = 2k} \\
\cos(\omega_k.t), & \mbox{if }i\mbox{ = 2k+1}
\end{cases}
$$

$$\omega_k = \frac{1}{10000^{2k/d}}$$  

이러한 함수의 정의에 따르자면, vector의 dimension에 따라 frequency가 줄어든다.(sin/cos의 주기가 길어짐)  
따라서 파장에서 $2\pi$에서 $10000*2\pi$까지의 기하 수열을 보인다.  
또한 positional embedding으로 들어가는 $\vec{p_t}$에 대해 각 frequency에서 sin,cos 쌍이 들어가 있음을 알 수 있다.(d는 2의 배수)  

$$
\vec{p_t} = 
\begin{bmatrix}
\sin(\omega_1.t) \\
\cos(\omega_1.t) \\
\\
\\
\sin(\omega_2.t) \\
\cos(\omega_2.t) \\
\\
\\
\vdots
\\
\\
\sin(\omega_{d/2}.t) \\
\cos(\omega_{d/2}.t) \\
\end{bmatrix}
_{d \times 1}
$$

위와 같이 $\vec{p_t}$는 cos과 sin의 순서쌍으로 그 position에 대한 정보를 담게 된다.  

## 3.Why Self-Attention  
self-attention이 RNN이나 Convolution보다 좋은지 3가지 양상으로 나누어 설명한다.  
![img1 daumcdn](https://github.com/sjh9824/NLP_Attention-is-all-you-need-review/assets/73771922/6508fe4e-deb1-4c25-82a0-ce15d7634947)  

#### 1)The total computational complexity per layer  
위 표에서 알 수 있듯이, self-attentino과 RNN을 비교했을 때 sequence lenth n이 representation dimensionality d보다 작아야지 complexity가 self-attention이 RNN보다 낮아짐을 볼 수 있다.  
하지만 보통의 경우 n이 d보다 작기 때문에 self-attetion의 complecity가 더 작다고 볼 수 있다.  

#### 2)The amount of computation that can be parallelized  
RNN은 input을 순차적으로 받아 학습 할 수 밖에 없다. 총 n번 RNN cell을 거치게 되는 반면, self-attention layer는 inpuy의 모든 position 값들을 연결하여 한번에 처리하기 때문에  
sequential operation이 O(1)을 가지게 되고 이는 parallel system에서 유리하게 사용된다.  

#### 3)The path length between long-range dependencies in the network    
long-range dependencies란 말그대로 position상 멀리 떨어져있는 단어들 간 dependency를 말하고 이를 학습하는 것은 sequence transduction task에서 key challenge에 해당된다.  
이러한 long-dependency를 잘 배우기 위해서 length of paths가 큰 영향을 미친다.  
![img1 daumcdn](https://github.com/sjh9824/NLP_Attention-is-all-you-need-review/assets/73771922/bfe6a251-d1e0-4a19-8d8f-84cc9dd71921)  
먼저 length of paths란 forward와 backward signals간의 길이를 말하며, 위의 그림을 예시로 path lengths는 한국어 tokens와 영어 tokens 길이를 말하는 것이다.  
또한 maximum path length는 I와 사랑해 사이의 길이 즉, encoder sequence length + decoder sequence length = 6이 된다.  
input과 output sequence 사이 조합 간 paths가 짧을수록 long-range dependencies를 더 잘 학습할 수 있고 따라서 논문에서 이러한 maximum path lengths도 비교하여 self_attention이 더 좋음을 증명한다.  
self-attention은 각 token들을 모든 token들과 참조하여 그 correlation information을 구해서 더해주기 때문에 maximum path length를 O(1)이라고 볼 수 있다.  

## 4.Experiments  
논문에서 진행한 실험에 대해 간단하게 살펴보았다.  
### 4.1 Machine Translation  
![img1 daumcdn](https://github.com/sjh9824/NLP_Attention-is-all-you-need-review/assets/73771922/6058073f-bf07-48ef-af34-48fb6f4349ec)  
먼저 English to German Translation task에 대해서 다른 모델과의 성능 비교표 이다.  
BLEU는 기계 번역 결과와 사람 번역 결과의 유사도를 비교하는 측정 방법이다.  
표에서 확인 할 수 있듯이, 다른 모델에 비해 Transformer가 제일 높은 성능을 보임과 동시에 training cost 또한 낮은 것을 확인 할 수 있었다.  
  
### 4.2 Model Variation  
<img width="568" alt="img1 daumcdn" src="https://github.com/sjh9824/NLP_Attention-is-all-you-need-review/assets/73771922/ec60abda-4181-4314-accf-160e56b83e42">  
모델의 여러 조건들을 변경해가면서 성능에 어떠한 영향을 주는지 확인하였다.  
(B)에서 key size인 $d_k$를 너무 줄이면 quality가 안좋아지며 (C)에서 큰 모델이 더 성능이 좋으며, (D) drop-out이 오버피팅을 피하는데 도움이 된다는 것을 확인 할 수 있었다.  

### 4.3 English Constituency Parsing  
<img width="466" alt="img1 daumcdn" src="https://github.com/sjh9824/NLP_Attention-is-all-you-need-review/assets/73771922/e28456f3-56ea-4acb-87a1-3cf58a5e2dd4">  
Transformer가 다른 task에서도 잘 동작하는지 보기 위해 English Constituency Parsing task에도 적용해본 결과이다.  
해당 Task는 어떤 단어가 문법적으로 어디에 속하는지 분류하는 Task로 해당 Task에 맞게 tuning 하지 않았음에도 좋은 성능을 보인다.  


## 5.마치며  
NLP에 무작정 관심이 있고 공부하고 싶다는 패기로 대학원에 들어와 첫 리뷰해본 논문이다.  
이 논문이 얼마나 대단한지 어떤 원리인지도 모른채 무작정 논문, 구글, 유튜브 등 몇일동안 서칭하면서 이해하는데 이해가 조금씩 되면서 이런 생각을 하는게 참 대단해보인다.  
나도 저렇게 연구할 수 있도록 더 공부하고 연구해야겠다는 생각이든다.  

사실 완벽히 다 이해한것은 아니지만 이제 지금까지 이해한것을 바탕으로 직접 코드로 구현해보면서 미처 이해하지 못한 부분이나 발견하지 못한 부분에 대해 학습하려고 한다.  
