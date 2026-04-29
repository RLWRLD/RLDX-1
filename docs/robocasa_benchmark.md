# RoboCasa Benchmark — 재현 가이드

이 문서는 RLDX의 **RoboCasa kitchen 벤치마크**를 재현하는 방법을 정리한다.
어떤 시나리오에서 어떤 스크립트를 돌려야 하고, 결과가 대략 몇이 나와야
하는지 (green / investigate / hard-block 밴드 포함) 일관된 기준을 제공한다.

벤치마크 메커닉 자체 (server/client 아키텍처, 공유 플래그, 일반 troubleshooting)는
[`docs/evaluation.md`](evaluation.md)에 정리되어 있으니 중복하지 않는다.
이 문서는 **"어떤 체크포인트로 어떤 숫자가 나와야 하는가"** 에 집중.

---

## 벤치마크 스펙 — RoboCasa Kitchen (24 task)

| 항목 | 값 |
|---|---|
| Simulator | robosuite + robocasa 0.1.x (venv: `rldx/eval/sim/robocasa/robocasa_uv/.venv`) |
| Embodiment | `GENERAL_EMBODIMENT` (`robocasa_panda_omron/*_PandaOmron_Env`) |
| Task 수 | **24** — kitchen scenes |
| 1 task당 episode 수 | **10** (일상 검증) / **50** (release seal) |
| Max episode steps | 720 |
| Action 실행 horizon | 16 |
| GPU | **4장 권장** (task 6개씩 shard) |
| 전체 runtime | 10 ep: **~20–25 min**, 50 ep: **~1 h 35 m** |

### 24 task list

```
TurnSinkSpout       TurnOnStove          TurnOnSinkFaucet    TurnOnMicrowave
TurnOffStove        TurnOffSinkFaucet    TurnOffMicrowave
PnPStoveToCounter   PnPSinkToCounter     PnPMicrowaveToCounter
PnPCounterToStove   PnPCounterToSink     PnPCounterToMicrowave
PnPCounterToCab     PnPCabToCounter
OpenSingleDoor      OpenDrawer           OpenDoubleDoor
CoffeeSetupMug      CoffeeServeMug       CoffeePressButton
CloseSingleDoor     CloseDrawer          CloseDoubleDoor
```

### 합격 밴드

`RLWRLD/RLDX-1-FT-ROBOCASA` 공개 체크포인트 기준으로
10 ep 벤치마크의 reference 성능은 **~68%**. Flow-matching 샘플링의
비-결정성 때문에 ±pt 편차가 자연스럽게 발생한다:

| 범위 | 판정 |
|---|---|
| ±3pt 이내 (65 – 71%) | **🟢 green** — 정상 변동 |
| ±3–5pt (63 – 73%) | **🟡 investigate** — log 확인, 재현 시도 |
| ±5pt 초과 | **🔴 hard-block** — 코드 회귀 가능성, merge 금지 |

50 ep release seal 기준점은 **~67.7%**. (에피소드 많을수록 variance 감소.)

---

## 1. 공개 체크포인트 로드 → eval

> **목적**: 코드를 건드리지 않고, 기존에 잘 동작하던 체크포인트를
> 불러와서 현재 브랜치의 **inference 경로가 멀쩡한지** 확인.
> Refactoring 도중 가장 자주 돌리는 검증이다.

### 스크립트

```bash
bash scripts/golden/eval_phase0.sh \
    RLWRLD/RLDX-1-FT-ROBOCASA \
    tests/golden/eval_runs/<run_label> \
    10
```

인자:
1. **모델 경로** — HF repo id 또는 로컬 checkpoint 디렉토리.
2. **출력 루트** — task별 `.mp4` + `summary.txt`를 쓸 디렉토리.
3. **episode 개수** — 기본 10. release seal 용은 50.

### 실행 환경

- 4 GPU 필요 (각 shard GPU 하나씩, port 20100–20103 자동 할당).
- **로그인 노드 금지 — worker node에서만 실행.**
- 환경변수 `NO_ALBUMENTATIONS_UPDATE=1` 은 launcher가 자동 export.

### 내부 동작

`eval_phase0.sh` 는 4개의 shard를 병렬로 띄운다. 각 shard는:

1. GPU `$gpu_id` 에 `run_rldx_server.py` 로 model server 시작 (port `20100 + gpu_id`).
2. 서버 ready 확인 후 할당된 6개 task를 순차 실행.
3. 각 task 당 rollout 은 `rldx/eval/rollout_policy.py` (robocasa venv) 로 실행.
4. 결과: `<run_label>/<TaskName>/summary.txt` + `.mp4` 여러 개.

4 shards × 6 tasks = 24 task 전체 커버.

### 기대 결과 (10 episodes/task, 240 total)

```
Aggregate: ~67.50% (162/240)   — green band
Per-task 성공률 0.20 – 1.00 범위로 분포
```

**정상이면 67–68% 대.** 65% 이하면 code regression 의심, 72% 이상이면
measurement error 의심 (flow matching variance 치고 너무 큼).

### 결과 집계

완료 후 task별 `summary.txt`를 합산:

```bash
python3 -c "
import os
root = 'tests/golden/eval_runs/<run_label>'
total = success = 0
for task in sorted(os.listdir(root)):
    sf = os.path.join(root, task, 'summary.txt')
    if not os.path.isfile(sf): continue
    with open(sf) as f:
        for line in f:
            if 'Success rate' in line:
                rate = float(line.split()[-1])
                total += 10; success += int(rate * 10)
                print(f'{task:30s} {rate:.2f}')
print(f'---\nTotal: {success}/{total} = {success/total*100:.2f}%')
"
```

결과를 `tests/golden/eval_runs/<run_label>/AGGREGATE_SUMMARY.md` 에
기록하고 commit (템플릿은 기존 run들 참고).

---

## 2. Pre-trained → RoboCasa finetune → eval

> **목적**: 현재 브랜치의 **학습 경로까지 전부** 검증. pre-trained
> backbone에서 출발해서 60k step finetune 후 eval.

### 2.1 Finetune 실행

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    bash run_scripts/train/finetune_rldx_robocasa.sh
```

핵심 설정 ([`run_scripts/train/finetune_rldx_robocasa.sh`](../run_scripts/train/finetune_rldx_robocasa.sh)):

| 설정 | 값 |
|---|---|
| Base model | `$BASE_MODEL_PATH` (스크립트 상단에서 수정) |
| Dataset | `$DATA_DIR` (required env; the `robocasa_mg_gr00t_300` set — 300 demos × 24 task) |
| Modality config | `rldx/configs/data/robocasa_config.py` → `Embodiment.GENERAL_EMBODIMENT` |
| GPU | 8장 기본 (`NUM_GPUS` 환경변수로 override) |
| Global batch | 64 |
| Max steps | **60,000** |
| Save steps | 1,000 (`save_total_limit=5`) |
| Learning rate | 4e-4 |
| Action horizon | 16 (`ROBOCASA.action_horizon`) |
| Video length | 4 (context window) |
| Color jitter | brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 |
| WandB project | `rldx-finetune` |
| Output | `ckpt/rldx/finetuned/rldx_ft_robocasa300_bsz64_60k/` |

예상 소요: 8×A100-SXM4-80GB 기준 **~24 h wall-clock**.

### 2.2 학습 도중 확인할 것

- WandB dashboard — loss 가 step 1k 이내에 급락 후 완만히 하락해야
  정상. Flat line 이면 dataset/modality config 연결 실패 의심.
- `ckpt/rldx/finetuned/.../checkpoint-{N}/` 이 1k step마다 생성되는지.
- `checkpoint-{N}/processor/processor_config.json` 가 정상 생성되는지
  (eval 시 이 파일을 읽음).

### 2.3 Smoke test (학습 경로 sanity check)

Full 24h run 전에 "학습이 일단 돌고 checkpoint가 저장되는지" 확인:

```bash
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
    bash run_scripts/train/finetune_rldx_robocasa.sh \
        --max-steps 10 \
        --save-steps 10 \
        --global-batch-size 8
```

스크립트 인자 override 는 `scripts/train.py` CLI 로 그대로 들어간다.
10-step smoke test 가 깨지면 60k full run 은 돌릴 가치 없음.

### 2.4 Finetune 완료 후 eval

```bash
bash scripts/golden/eval_phase0.sh \
    ckpt/rldx/finetuned/rldx_ft_robocasa300_bsz64_60k/checkpoint-60000 \
    tests/golden/eval_runs/<run_label>_ft60k \
    10
```

**주의**: `--model-path` 에는 `{run_dir}/checkpoint-{step}/` 서브
디렉토리를 넘긴다. `processor_config.json` 은 그 아래 `processor/` 에
있고, `rldx_policy` 는 이 layout 을 기대함.

### 2.5 기대 결과

동일한 base model / dataset / config 로 다시 finetune 하면 **공개
체크포인트와 같은 밴드 (67–68% ±3pt)** 가 나와야 한다. 벗어나는
경우:

| 구간 | 판정 | 의심 지점 |
|---|---|---|
| 65 – 71% | 🟢 green | 정상 — pre-train/finetune/eval 경로 OK |
| 63 – 65% / 71 – 73% | 🟡 investigate | seed / hyperparameter 변동, 재현 시도 |
| < 63% 또는 > 73% | 🔴 hard-block | 학습 pipeline 회귀 또는 eval 경로 회귀 |

Step-wise eval (학습 중간 checkpoint eval) 은 수렴 곡선 확인용으로
optional:

```bash
for step in 10000 20000 30000 40000 50000 60000; do
    bash scripts/golden/eval_phase0.sh \
        ckpt/.../checkpoint-$step \
        tests/golden/eval_runs/ft_step_$step \
        10
done
```

일반적으로 30k 이후 plateau, 60k 에서 수렴.

---

## 3. 작은 검증 루프 (10 ep, 1 task)

> **목적**: 코드 변경 직후 action pipeline 이 일단 깨지지 않는지
> 확인. ~2 min, GPU 1장. Full 24-task eval 돌리기 전 최소한의 sanity.

### 스크립트

2 process split 이 필요해서 launcher 없이 직접 띄운다:

```bash
# Terminal 1 — model server
CUDA_VISIBLE_DEVICES=0 uv run python rldx/eval/run_rldx_server.py \
    --model-path RLWRLD/RLDX-1-FT-ROBOCASA \
    --embodiment-tag GENERAL_EMBODIMENT \
    --use-sim-policy-wrapper \
    --host 127.0.0.1 --port 20100

# Terminal 2 — rollout (1 task, 10 ep)
rldx/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
    rldx/eval/rollout_policy.py \
        --n_episodes 10 \
        --policy_client_host 127.0.0.1 --policy_client_port 20100 \
        --max_episode_steps 720 \
        --env_name "robocasa_panda_omron/TurnSinkSpout_PandaOmron_Env" \
        --n_action_steps 16 --n_envs 1 \
        --video_dir /tmp/smoke_TurnSinkSpout
```

### 기대 결과

`TurnSinkSpout` 는 쉬운 task라 healthy 한 run 이면 **7–9/10 success**.
1–2/10 이면 거의 확실히 코드 regression.

### 확인 포인트

- 서버 로딩 시간: 30–60 s (backbone 로드 + flash-attn 초기화). 이
  동안 `Connection refused` 는 정상.
- Rollout 시작 후 첫 episode 완료까지: ~10 s.
- 10 episode 전체: ~1–2 min.
- `/tmp/smoke_TurnSinkSpout/` 에 `episode_*-success.mp4` 또는
  `-failure.mp4` 생성.

---

## 부록 — 결과 커밋 convention

각 run 은 고유 디렉토리 하나에 격리:

```
tests/golden/eval_runs/<label>/
├── AGGREGATE_SUMMARY.md        ← 수동 작성, 요약 + 비교표
├── _launcher_logs/
│   ├── shard-gpu0.log
│   ├── shard-gpu1.log
│   ├── shard-gpu2.log
│   └── shard-gpu3.log
└── <Task>/
    ├── eval.log
    ├── simulation_results.csv
    ├── summary.txt
    └── robocasa_panda_omron_<Task>_PandaOmron_Env_env00-episode_*-{success,failure}.mp4
```

`AGGREGATE_SUMMARY.md` 에 포함할 것:
- HEAD commit SHA
- 사용한 체크포인트 (HF repo id 또는 local path)
- N_EPISODES 및 task 수
- Aggregate success % (분자/분모)
- 이전 run 들과의 비교 (같은 체크포인트 기준)
- green / investigate / hard-block 판정

---

## 부록 — Troubleshooting (benchmark-specific)

일반적인 문제 (서버 로딩 타이밍, `HF_HOME`, resume) 는
[`docs/evaluation.md` Troubleshooting](evaluation.md) 참조. 여기선
RoboCasa 특이 사항만:

### 모든 task 가 `exit=1` 로 실패

서버 로드는 성공했는데 `rollout_policy.py` 가 `AttributeError` 나
`TypeError` 로 즉시 종료하는 경우. 대부분 **policy의 RobotConfig /
processor 인터페이스와 `rollout_policy.py` 의 기대 shape 불일치**.
`_launcher_logs/shard-gpu*.log` 와 `<Task>/eval.log` 둘 다 확인.

대표 패턴:
- `RLDXPolicy object has no attribute 'get_modality_config'` —
  `rldx/policy/serving/rldx_policy.py` 에 method alias 누락.
- `Server error: <ZMQ decode failure>` — `get_modality_config` 의
  반환값이 plain dict 가 아님. rollout venv에 rldx 가 안 깔려있어서
  커스텀 클래스 pickle 역직렬화 실패.

### 특정 task 만 체계적으로 0%

대부분 robocasa asset mismatch (demo data 와 sim env 사이) 또는
embodiment 별 action mapping 회귀. `CoffeeSetupMug` /
`PnPCabToCounter` / `TurnOffStove` 가 20% 이하로 떨어지는 건 reference
run 에서도 관찰되는 "어려운 task" 라 변동 범위로 간주 OK.

### Resume

`rollout_policy.py` 는 video_dir 의 기존 `episode_*.mp4` 를 스캔해
다음 번호부터 이어 씀. 중단된 run 은 같은 명령어 재실행으로 resume
가능. Task 단위 resume 은 안 되므로 해당 task 디렉토리 삭제 후 재실행.
