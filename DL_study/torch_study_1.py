# 코드 설명
"""
임의의 x, y를 batch size(64) 개 만큼 추출하고, 그 배치에 대해서 반복해서 오차를 줄이는 방식 테스트
즉, 데이터 개수가 그냥 64 개라고 생각하면 됨
배치 하나에 대해서만 실행됨
"""

import torch as th


DEVICE = th.device('cuda' if th.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
INPUT_SIZE = 1000
HIDDEN_SIZE = 100
OUTPUT_SIZE = 10

x = th.randn(BATCH_SIZE,
            INPUT_SIZE,
            device=DEVICE,
            dtype=th.float,
            requires_grad=False)

y = th.randn(BATCH_SIZE,
            OUTPUT_SIZE,
            device=DEVICE,
            dtype=th.float,
            requires_grad=False)

w1 = th.randn(INPUT_SIZE,
            HIDDEN_SIZE,
            device=DEVICE,
            dtype=th.float,
            requires_grad=True)

w2 = th.randn(HIDDEN_SIZE,
            OUTPUT_SIZE,
            device=DEVICE,
            dtype=th.float,
            requires_grad=True)

learning_rate = 1e-6

for t in range(1,501):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    loss = (y_pred -y).pow(2).sum()
    loss.backward()

    if t % 100 == 0:
        print("Iteration: ", t, "\t", "Loss: ", loss.item())


    with th.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
    


        w1.grad.zero_()
        w2.grad.zero_()