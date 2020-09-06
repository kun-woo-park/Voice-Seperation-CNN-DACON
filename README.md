# Voice-Seperation-CNN

## 구현 목적
데이콘에서 진행되었던 음성중첩 분리 학습을 위해 진행되었다. 총 10만개의 음성데이터를 이용하여 학습하고, 이 10만개의 음성 데이터 내에 어떤 단어들이 있는지 구분해 내는것이 목적이다.
해당 데이터는 다음 링크에서 받을 수 있다. https://drive.google.com/file/d/1-v8Uc_CiTLRbAeP_k8nYimDdYcmlhF1c/view?usp=sharing

## 구현 내용
음성 중첩된 데이터들을 가공하는 방법으로 Mel spectogram을 사용하였다. Mel spectogram을 형성하는데는 torchaudio의 함수를 사용하였다. 사용한 factor는 다음 코드와 같다.

네트워크는 CNN의 형식을 채택했으며, 네트워크는 다음과 같이 구현하였다.

Batch size = 256, Epoch = 100으로 설정하여 학습을 진행하였다.

## 구현 결과
결과를 보면, 9 Epoch 즈음까지 Valloss가 수렴하다가, 이후 발산하는 양상을 볼 수 있다. 따라서 9 Epoch 까지는 Well-fitted 되어있었으나 이후 overfitting이 발생했다고 예측한다. 해당 모델에 대한 데이콘 결과는 loss 1.36987가 나왔고, 총 396팀중 34등으로 순위권내에는 위치하지 못하였으나, 첫 데이콘 도전에 의의가 있다고 생각된다.
