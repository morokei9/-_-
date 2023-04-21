# -_-
OOO 교수님 안녕하세요 🤣 진짜 보잘것 없긴 하지만 그래도 입시설명회 때 깃허브에 잇는 거 보시면 좋을 것 같다고 하셔서 한번 올려봅니다.


정말 보잘것 없긴 한데, 그래도 남한테 의뢰한 거 아니고 직접 만든 거라서 말씀 참고해서 한번 깃허브에 올려봅니다.

04.13.2023


이름이나 이런 건 문제 될 것 같아서 쓰지 않겠습니다.

MIT LICENSE

MIT LICENSE 써서, 잘 되는 거 말고, 형태랑 의미는 맞는데 CNN 구조 및 DQN 일부가 최적화 되지 않거나 이상해서 잘 안되는 코드 일부러 올렸습니다 🤣🤣






BGM 에 있는 JAZZ MUSIC 1~7 은 COPYRIGHT FREE MUSIC 입니다.





BETAZERO 는 DQN (Deep Q (value) Network) Reinforcement Learning Algorythm 과 CNN (Convolutional Neural Network) 를 사용해서 스스로 일반룰의 체스를 플레이하는 프로그램입니다.

CNN 은

체스보드가
14x8x8 (기물의 흰 백 정보 1 + 각 기물 흰 백 / 룩 폰 킹 퀸 나이트 비숍 13개)
로 변환되어 인풋을 받게 되고,

알파제로를 참고해서
Residual Network + BAtch Normalization 을 사용한 Hidden layers 가 있습니다.

제 그래픽 카드가 RTX 3070 이라서 히든 레이어 CNN 2개에 Linear 1개 했습니다.......ㅠ



DQN 은

Hyperparameters 들은 Optuna 를 사용하여 Optimization 을 돌렸는데,

Replay Buffer Size 는 우선은 1000000 개로 고정 해놨고 

BATCH 도 1000개정도만 했습니다.


Q Function  의 경우 :

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.local_net(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.local_net, self.target_net, self.tau)

크게 특징 적인 것은 없습니다.

DQN CNN 에 들어가고 나오는 Chess.Board 와 Chess.Move 에 대한 인코딩과 디코딩을 스스로 했다는 점 ? 은 있는 것 같습니다.





현재까지는 STOCKFISH 등의 대적 상대 없이 오직 폐관수련으로만 연습 하고 있습니다.



Python 3.10 / Jupyter Notebook

Torch, Gym
Optuna

** Python - Chess ** 모듈의 도움을 많이 받았습니다. 이 모듈과 Gym-Chess module 이 없었다면, 룰을 코딩하는 시점에서 진작에 포기 했을 것 같습니다.



np pd sklearn 등 기본 모듈도 사용했습니다.

OpenAI 의 GPT-4 의 힘을 많이 빌렸습니다.



BETAZERO의 이름은 대놓고 Google Deepmind 의 AlphaZero 에서 따왔습니다. 저작권에 걸리진 않겠죠 ? 




폐관수련 한 기준으로 모델의 ELO 가 2700을 넘게 되면

이후 Variation 으로는
1) CLOWNPUSHER (모든 오프닝 BONGCLOUD)
2) 패작체스 (Anti-Chess) AI
3) Inversion Chess (Tenet) AI


등에도 기본 구조를 사용할 예정입니다.

에 모델이 바보일 때 부터 고수일 때까지 기보를 섬섬히 넣었습니다.
