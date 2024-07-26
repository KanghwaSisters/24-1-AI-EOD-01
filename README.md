# 24-1-AI-EOD-01
[ 24-1 /  AI EOD / Team 01 ]  
👩‍💻 이승연, 변지은

# 목차

1. [Environment](#-environment)
    1. [State](##-state)
    2. [Reward Design](##-reward-design)
    3. [속도 개선](##-속도-개선)
    4. [Render](##-render)



---
# 1. Environment
## State

- (`nrow`, `ncol`) 차원의 행렬 (type: `np.ndarray`)

| 상황 | 표현 |
| --- | --- |
| 지뢰 | -2 |
| 가려진 타일 | -1 |
| 그 외 | 주변 지뢰의 개수를 나타내는 숫자 (0~8) |

![environment1](environment1.png)

<br>
<br>

- - -

## Reward Design

### reward 조건

- 지뢰 `mine`: 지뢰를 밟은 경우
- 성공 `clear`: 지뢰를 제외한 모든 좌표가 열린 경우
- 중복 행동 `overlapped`: 상태 맵에서 이미 열린 행동을 선택하는 경우
- 추측 행동 `guess`: 주변에 열린 좌표가 없는데 선택한 경우 (내장 함수로 판단)
- 좋은 행동 `empty`: 유효한 행동. 추측이나 중복이 아니고 지뢰가 아닌 행동

![environment2](environment2.png)

<br>
<br>

### 양수 보상 체계

|  | mine | clear | empty | overlapped | guess |
| --- | --- | --- | --- | --- | --- |
| reward | 0 | 0 | 1 | 0 | 0 |
| done | True | True | False | True | False |

- 양수 보상 체계에서는 “중복 선택 시 게임 종료” 조건이 필수적이다. 보상을 통해 중복 행동에서 빠져나오기 매우 어려운 구조이기 때문이다.

- 초반 낮은 성능에서 어느정도 수렴할 때 까지 가파른 기울기로 상승한다.

- 양수 보상 체계를 사용한 이유
    - 좋은 행동을 할 때만 양수의 보상 → 보상을 최대화 하도록 학습 → 성공률 상승
    - 에이전트가 좋은 행동만 하도록 학습하면 자연스럽게 성공률이 오를 것이라고 생각했다.

- 간과한 점
    - 클리어를 할 때의 이동 횟수가 모두 같지 않음. 즉, 보상이 높다고 클리어하는 것도 아니고 보상이 낮다고 클리어하지 않는 것도 아니다.
    - 클리어하는 경우에 보상을 추가로 더 크게 주는 것으로 위의 문제 상쇄 시도
    → 하지만 이동 횟수의 차이로 인해 최대 보상의 편차가 존재한다.
    - 결정적으로 “중복 선택 시 게임 종료”라는 조건이 학습 속도를 너무 느리게 하고, 지뢰찾기 게임과도 맞지 않는 조건이다.

<br>
<br>

### 음수 보상 체계

|  | mine | clear | empty | overlapped | guess |
| --- | --- | --- | --- | --- | --- |
| reward | -1 | 1 | 1 | -1 | 0.3 |
| done | True | True | False | False | False |

- 학습 초반부터 일정한 기울기로 성능이 향상한다.

- 지뢰의 보상을 아주 낮게 주는 것은 의미가 없다. 어차피 지뢰를 밟으면 끝이기 때문이고, 에이전트가 학습할 때 극단값이 생겨 학습에 혼돈이 생긴다. 게임 종료 시의 보상보다는 “좋은 행동”을 많이 하도록 유도하는 것이 핵심이다.

- 보상들의 비율
    - 경험 상 지뢰와 클리어의 보상은 비율을 맞추는 것이 좋다.
    - 하지만 게임을 진행하는 동안 발생하는 중복, 좋은 행동, 추측 보상의 비율을 맞추기보다는 중복 행동이 학습에 있어서 가장 큰 문제점이기 때문에 지뢰와 같은 가장 낮은 보상을 주었다.
    - 좋은 행동과 클리어는 모두 유도해야할 행동이므로 가장 큰 보상을 주었다.
    - 추측한 행동의 경우 지뢰나 중복보다는 나은 행동이고, 추측한 행동을 통해 운이 좋게 판이 열릴 수도 있기 때문에 작은 양수의 보상을 주었다.

- 클리어 보상의 크기가 너무 크면 안된다고 생각한다.(사고실험)
    - 클리어 보상의 크기가 다른 보상보다도 너무 큰 경우, 다른 어떤 행동을 해서라도 클리어만 하면 보상이 커지기 때문에 효율적으로 움직이지 않을 가능성이 있다. 즉, 좋은 행동을 할 유도가 작아진다.

- 전체적으로 보상의 크기가 크면 학습이 불안정하다. 따라서 모두 -1~1 사이의 값으로 설정했다.

<br>
<br>

- - -

## 속도 개선

- list 자료형 대신 np.array 자료형 사용
- for문 최소화 → numpy 함수 활용
- action: index (0~`nrow*ncol`-1)
    - `divmod()` 함수를 통해 좌표로 바꿔서 사용

<br>
<br>

- - -

## Render

- pandas.DataFrame으로 맵 시각화
- `render_answer()`, `render(state)` 함수로 구현
- `render_color()` 함수로 숫자별 색 적용

![environment3](environment3.png)
