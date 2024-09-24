# Entropy
크로스 엔트로피를 알기전에 엔트로피가 뭔지 부터 알아야 한다.   
엔트로피는 간단히 말하면 불확실성을 나타내는것이고, 만약 엔트로피가 높으면 해당 데이터에   
대해 예측이 어렵다는 뜻이다.   

$$
H(x) = -\sum_{i=1}^{n} p(x_i) \log p(x_i)
$$

## 간단한 예시 (주사위 던지기 vs 동전 던지기)
동전을 던졌을때 앞/뒷면이 나올 확률을 예측해보자 !  
$$ 
H(x) = - \left( \frac{1}{2} \log \frac{1}{2} + \frac{1}{2} \log \frac{1}{2} \right)
$$

계산 결과 대략 0.693이 나온다. 여기서 $log$는 $\log_{exponential}$.   

---

그러면 주사위를 던져 각 6면의 확률을 계산해보자 !  
$$ H(x) = - \left( \frac{1}{6} \log \frac{1}{6} + \frac{1}{6} \log \frac{1}{6} + \frac{1}{6} \log \frac{1}{6} + \frac{1}{6} \log \frac{1}{6} + \frac{1}{6} \log \frac{1}{6} + \frac{1}{6} \log \frac{1}{6} \right)
$$
 
계산 결과 대략 1.79가 나온다.   

위에서 볼 수 있다 싶히 주사위의 엔트로피 값이 동전보다 높다.   
즉 주사위 결과에 대한 예측이 동전보다 휠씬 어렵다는것을 알 수 있다.   

❗❗ 결과적으론 $\log x$ 에서 $x$ 값이 작으면 작을 수록 $log$ 값이 작지만 마지막에 마이너스가 붙기에 값이 커진다.

---

# Cross entropy
크로스 엔트로피는 q 분포를 예측하는게 주 목적이다, 다시말해 q를 예측하는 p 분포가 만들어졌을때 얼마나 같은지 확인하는것이다.    

$$
H_p(q) = -\sum_{i=1}^{n} q(x_i) \log p(x_i)
$$

예를 들어 상자안에는 0.8/0.1/0.1의 비율로, 빨간/초록/파랑 공이 들어가 있다고 하자.   
하지만 A 유저의 직감으로는 0.2/0.2/0.6인것 같다,   
다른 B 유저의 직감으로는 0/7/0.1/0.2인것 같다,   
이때 cross-entropy 결과값은 이하와 같다.

$Entropy :$
$$
H(q) = -[0.8 \log(0.8) + 0.1 \log(0.1) + 0.1 \log(0.1)] = 0.63
$$

## A 유저
$Cross-entropy :$
$$
H_p(q) = -[0.8 \log(0.2) + 0.1 \log(0.2) + 0.1 \log(0.6)] = 1.50
$$

## B 유저
$Cross-entropy :$
$$
H_p(q) = -[0.8 \log(0.7) + 0.1 \log(0.1) + 0.1 \log(0.2)] = 0.68
$$

알 수 있다 싶히 B 유저의 값이 정답 값이랑 매우 근사하다.

# Kullback-Leibler Divergence
entropy와 cross-entropy를 알았다면 KL-divergence도 알아야한다 !!
이의 개념도 cross-entropy와 매우 흡사하다, 즉 서로 다른 두 분포의 차이를 측정하는데 쓰인다.   
바로 수식을 보자 !   

$$
D_{KL}(p \parallel q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}
$$

여기서 $q(x)$ 는 항상 고정이고 $q(x)$ 는 $p(x)$보다 항상 크다. 왜냐하면 우리는 $p(x)$ 해당 분포를 $q(x)$와 같게 해야하는게 주 목적이다. 다시말해 $q(x)$ 분포와 $p(x)$ 가 비슷하면 비슷할 수록 $KL-divergence$ 는 0에 가까워지게 된다.


### Reference
[Entropy, Cross-entropy. KL-divergence](https://velog.io/@rcchun/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%ED%81%AC%EB%A1%9C%EC%8A%A4-%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BCcross-entropy)