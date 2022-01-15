# LSTM

## RNN

출처: [개발새발로그](https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr) 

전통적인 neural network이 이렇게 지속되는 생각을 하지 못한다는 것이 큰 단점이다. 예를 들어, 영화의 매 순간 일어나는 사건을 분류하고 싶다고 해보자. 전통적인 neural network는 이전에 일어난 사건을 바탕으로 나중에 일어나는 사건을 생각하지 못한다.

Recurrent neural network (이하 RNN)는 이 문제를 해결하고자 하는 모델이다. RNN은 스스로를 반복하면서 이전 단계에서 얻은 정보가 지속되도록 한다.

![Recurrent neural networks have loops](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile26.uf.tistory.com%2Fimage%2F99B366365ACB86A014D902)

위 그림에서 A는 RNN의 한 덩어리이다. A는 input xtxt를 받아서 htht를 내보낸다. A를 둘러싼 반복은 다음 단계에서의 network가 이전 단계의 정보를 받는다는 것을 보여준다.

이러한 RNN의 반복 구조는 혹여 불가사의해 보일 수도 있겠지만, 조금 더 생각해보면 RNN은 기존 neural network와 그렇게 다르지 않다는 것을 알 수 있다. RNN을 하나의 network를 계속 복사해서 순서대로 정보를 전달하는 network라고 생각하는 것이다. 아예 반복을 풀어버리면 좀 더 이해하기 쉬울 것이다.

![image.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile3.uf.tistory.com%2Fimage%2F9901A1415ACB86A0211095) 

이렇게 RNN의 체인처럼 이어지는 성질은 곧장 sequence나 list로 이어지는 것을 알려준다. 이런 데이터를 다루기에 최적화된 구조의 neural network인 것이다.

## LSTM(Long Short-Term Memory Network)

RNN의 성공의 열쇠는 "Long Short-Term Memory Network" (이하 LSTM)의 사용이다. LSTM은 RNN의 굉장히 특별한 종류로, 아까 얘기했던 영화를 frame 별로 이해하는 것과 같은 문제들을 단순 RNN보다 정말 훨씬 진짜 잘 해결한다. 기존 RNN도 LSTM만큼 이런 일을 잘 할 수 있다면 RNN은 대단히 유용할텐데, 아쉽게도 RNN은 그 성능이 상황에 따라 그 때 그 때 다르다.

우리가 현재 시점의 뭔가를 얻기 위해서 멀지 않은 최근의 정보만 필요로 할 때도 있다. 예를 들어 이전 단어들을 토대로 다음에 올 단어를 예측하는 언어 모델을 생각해 보자. 만약 우리가 "the clouds are in the *sky*"에서의 마지막 단어를 맞추고 싶다면, 저 문장 말고는 더 볼 필요도 없다. 마지막 단어는 sky일 것이 분명하다. 이 경우처럼 필요한 정보를 얻기 위한 시간 격차가 크지 않다면, RNN도 지난 정보를 바탕으로 학습할 수 있다.

하지만 반대로 더 많은 문맥을 필요로 하는 경우도 있다. "I grew up in France... I speak fluent *French*"라는 문단의 마지막 단어를 맞추고 싶다고 생각해보자. 최근 몇몇 단어를 봤을 때 아마도 언어에 대한 단어가 와야 될 것이라 생각할 수는 있지만, 어떤 나라 언어인지 알기 위해서는 프랑스에 대한 문맥을 훨씬 뒤에서 찾아봐야 한다. **이렇게 되면 필요한 정보를 얻기 위한 시간 격차는 굉장히 커지게 된다.**

안타깝게도 이 격차가 늘어날 수록 RNN은 학습하는 정보를 계속 이어나가기 힘들어한다.

이론적으로는 RNN이 이러한 "긴 기간의 의존성(long-term dependencies)"를 완벽하게 다룰 수 있다고 한다. 그리고 단순한 예제에 대해서는 사람이 신중하게 parameter를 골라서 그 문제를 해결할 수도 있다. 하지만 RNN은 실제 문제를 해결하지 못 하는 것이 슬픈 현실이다. 이 사안에 대해 [Hochreiter (1991)](http://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf)과 [Bengio 외 (1994)](http://www-dsi.ing.unifi.it/~paolo/ps/tnn-94-gradient.pdf)가 심도있게 논의했는데, RNN이 긴 의존 기간의 문제를 어려워하는 꽤나 핵심적인 이유들을 찾아냈다.

LSTM은 RNN의 특별한 한 종류로, 긴 의존 기간을 필요로 하는 학습을 수행할 능력을 갖고 있다. LSTM은 [Hochreiter & Schmidhuber (1997)](http://www.bioinf.jku.at/publications/older/2604.pdf)에 의해 소개되었고, 그 후에 여러 추후 연구로 계속 발전하고 유명해졌다. LSTM은 여러 분야의 문제를 굉장히 잘 해결했고, 지금도 널리 사용되고 있다.

LSTM은 긴 의존 기간의 문제를 피하기 위해 명시적으로(explicitly) 설계되었다. 긴 시간 동안의 정보를 기억하는 것은 모델의 기본적인 행동이어야지, 모델이 그것을 배우기 위해서 몸부림치지 않도록 한 것이다!

모든 RNN은 neural network 모듈을 반복시키는 체인과 같은 형태를 하고 있다. 기본적인 RNN에서 이렇게 반복되는 모듈은 굉장히 단순한 구조를 가지고 있다. 예를 들어 tanh layer 한 층을 들 수 있다.

LSTM도 똑같이 체인과 같은 구조를 가지고 있지만, 각 반복 모듈은 다른 구조를 갖고 있다. 단순한 neural network layer 한 층 대신에, 4개의 layer가 특별한 방식으로 서로 정보를 주고 받도록 되어 있다.

![image.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile30.uf.tistory.com%2Fimage%2F999F603E5ACB86A00550F0)

![image.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile5.uf.tistory.com%2Fimage%2F993A93495ACB86A02FFAA8)

위 그림에서 각 선(line)은 한 노드의 output을 다른 노드의 input으로 vector 전체를 보내는 흐름을 나타낸다. 분홍색 동그라미는 vector 합과 같은 pointwise operation을 나타낸다. 노란색 박스는 학습된 neural network layer다. 합쳐지는 선은 concatenation을 의미하고, 갈라지는 선은 정보를 복사해서 다른 쪽으로 보내는 fork를 의미한다.

### Main idea

LSTM의 핵심은 cell state인데, 모듈 그림에서 수평으로 그어진 윗 선에 해당한다.

Cell state는 컨베이어 벨트와 같아서, 작은 linear interaction만을 적용시키면서 전체 체인을 계속 구동시킨다. 정보가 전혀 바뀌지 않고 그대로 흐르게만 하는 것은 매우 쉽게 할 수 있다.

![image.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile10.uf.tistory.com%2Fimage%2F99CB87505ACB86A00FAB6F)

LSTM은 cell state에 뭔가를 더하거나 없앨 수 있는 능력이 있는데, 이 능력은 gate라고 불리는 구조에 의해서 조심스럽게 제어된다.

Gate는 정보가 전달될 수 있는 추가적인 방법으로, sigmoid layer와 pointwise 곱셈으로 이루어져 있다.

Sigmoid layer는 0과 1 사이의 숫자를 내보내는데, 이 값은 각 컴포넌트가 얼마나 정보를 전달해야 하는지에 대한 척도를 나타낸다. 그 값이 0이라면 "아무 것도 넘기지 말라"가 되고, 값이 1이라면 "모든 것을 넘겨드려라"가 된다.

LSTM은 3개의 gate를 가지고 있고, 이 문들은 cell state를 보호하고 제어한다.

.
.
.

나머지 부분은 출처 사이트 참고! 거의 다 수학임.

