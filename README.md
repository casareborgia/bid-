# 공공입찰 예정가격(사정률) 예측 모델

추정가격(또는 나라장터 공고번호) 하나를 입력하면 기초금액, 예상 예정가격 Q25/Q50/Q75, 낙찰하한선을 통과하는 기준별 투찰금액을 출력하는 로컬 도구. **개인 학습용 프로젝트**이며 예측은 참고용이다.
모델은 두 개다. **사내 모델**(사내 경쟁입찰 데이터, 예가 범위 ±2%, 데이터·모델 비공개)이 기본이고, **시장 모델**(나라장터 538,907건, 예가 범위 ±2%/±3%)은
`--market --range N` 또는 `--market --agency "발주기관명"` 으로 쓴다. 공공발주 전체에 적용할 때는 시장 모델을 쓴다.
도메인 규칙·작업 기록 문서는 사내 정보가 섞여 있어 저장소에 넣지 않는다.

> 공개 범위: 이 저장소에는 코드·규칙표·테스트(공개 API 응답 녹화본, 담당자 정보 마스킹)만 있다. 원본 계약 데이터·수집 데이터·모델 파일은 저장소에 없다(`.gitignore`).
> 공개 모델은 나라장터 데이터로만 학습한 시장 모델이며 허깅페이스 `casareborgia/yega-rate-predictor-market`에 둔다(`src/publish_hf.py --profile public-market`).
> 조달청 Open API 활용가이드(docx)는 저장소에 넣지 않는다. 공공데이터포털 「조달청_나라장터 낙찰정보서비스」·「조달청_나라장터 입찰공고정보서비스」 페이지에서 받는다.

## 준비

```bash
brew install libomp          # LightGBM 의존
uv sync                      # Python 3.12 가상환경 + 패키지
cp .env.example .env         # G2B_SERVICE_KEY 입력 (조달청 낙찰정보서비스 인증키)
```

사내 계약 데이터 CSV는 `data/raw/`에 둔다 (git 추적 안 함, 비공개).

## 실행 순서

```bash
# Phase 1. 사내 계약 데이터 정제 → data/internal_clean_post2019.parquet
uv run python src/preprocess_internal.py

# Phase 2. 나라장터 API 수집 (월 단위, 일일 950호출에서 자동 중단, 재실행 시 이어받기)
#   2026-10-27 완료: 2024-01~2026-09 33개월 전체 보유(539,631건).
uv run python src/collect_g2b_api.py --start 2024-01 --end 2026-09            # 공사+용역 전체 시장 (zsh out/collect_loop.sh 로 일별 반복)
uv run python src/collect_g2b_api.py --mode agency --agency 한국남동발전 --start 2024-01 --end 2026-09
uv run python src/collect_g2b_api.py --build                                  # 원시 jsonl → parquet

# Phase 3. 학습 + 5-Fold 시계열 교차검증 → out/cv_report.md, models/bidding_model.joblib
#   수의계약(contract_method "수의")은 자동 분리되어 경쟁입찰만 학습하고, 수의계약은 경험 분포로 모델에 저장된다.
uv run python src/train_model.py
uv run python src/train_model.py --compare              # 3a(사내)/3b(사내+나라장터) 비교 → out/cv_compare.md (모델 저장 없음)
uv run python src/train_model.py --with-g2b             # 나라장터 시장 행을 가중치(사내 3:시장 1)로 합쳐 학습 (3b 채택 시)

# Phase 6. 시장 모델 학습 (나라장터 538,907건, 예가 범위별 사전값) → out/cv_report_market.md, models/market_model.joblib
uv run python src/train_model.py --source market

# Phase 4. 예측 (사내 모델 기본, --market 이면 시장 모델)
uv run python src/predict_cli.py --est_price 1000000000 --work_type 2 --bidder_count 25 --lower_limit 87.745
uv run python src/predict_cli.py --est_price 1000000000 --work_type 1 --market --range 3   # 타 기관 입찰, 공고의 예가 범위 ±3%
uv run python src/predict_cli.py --est_price 1000000000 --work_type 1 --market --agency "서울특별시 강남구"   # 발주기관명으로 예가 범위(±3%)·하한율 자동 판정
uv run python src/predict_cli.py --bid_no R26BK01752065   # 나라장터 공고번호 하나로: 추정가격·기초금액·예가 범위·하한율·A값·발주기관 자동
uv run python src/bid_notice.py R26BK01752065            # 공고 조회값만 JSON 으로 확인
uv run python src/predict_cli.py --est_price 500000000 --award_method 협상 --dept OO본부 --json
uv run python src/predict_cli.py --est_price 500000000 --negotiated          # 수의계약: 경험 분포 Q25/Q50/Q75

# Phase 7 & 8. 실제값 검증 + LLM 인터페이스 + 계약 규정 감시
uv run python src/evaluate.py --bid_no 20240617004                                  # 공고번호로 실제 예정가격과 예측 비교 (수집 범위 밖이면 API 실시간 조회)
uv run python src/evaluate.py --est_price 1000000000 --actual_yega 1098000000 --work_type 1   # 수동 비교(사내 모델)
uv run python src/evaluate.py --holdout 2026-09 --source market --sample 10          # 컷오프 이전 재적합 → 홀드아웃 성적 (out/backtest_*.md)
uv run python src/evaluate.py --holdout 2026-01~2026-10 --source internal
uv run python src/ollama_chat.py --once "서울특별시 강남구 공사, 추정가격 10억. 투찰가?"     # Ollama gemma4:26b 자연어 (도구 호출)
uv run python src/mcp_server.py                                                      # MCP 서버 (Claude Code 는 .mcp.json 으로 자동 등록)
uv run python src/publish_hf.py --push                                              # 모델·리포트·모델 카드를 HF 비공개 저장소에 업로드 (기본은 --dry-run)
uv run python src/monitor_regime.py --month 2026-09 --skip-collect              # 예가 규정 변경 감시: 1개월 분포 점검 + 홀드아웃 (기본은 지난달 수집 후 점검)
uv run python src/monitor_rules.py                                              # 계약 법령·예규·기관 규정 변경 감시 (Phase 8, 변경 승인 시 --accept)
uv run python src/check_notice_rules.py --month 2026-10                         # 하한율·예가 범위 규칙표를 입찰공고 1개월로 대량 검증 (out/notice_rules_check_*.md)
```

테스트: `uv run pytest -q`

## 실제값 검증과 LLM 인터페이스 (Phase 7)

`evaluate.py`는 예측이 실제 예정가격과 얼마나 맞았는지 두 방식으로 본다.

- **단건**: 공고번호(`--bid_no`)로 수집 parquet 의 실제 기초금액·예정가격·예가 범위를 찾아 Q25/Q50/Q75 예측과 비교하고, 기준별 투찰가가 실제 낙찰하한을 넘었는지(유효/미달) 표시한다. 실제값을 직접 아는 건은 `--est_price --actual_yega`로 넣는다.
- **배치(홀드아웃)**: `--holdout 2026-09`처럼 월을 주면 그 이전 데이터로 모델을 다시 적합해 홀드아웃 월의 실제값과 비교한다(in-sample 이 아닌 정직한 성적).

| 홀드아웃 (2026-10-31) | n | Q25~Q75 커버리지 | Q75 투찰가 유효율 (마진 0 / 0.1%) | ML MAE vs Baseline |
|---|---|---|---|---|
| 시장 모델, 2026-09 (학습은 2026-08 이전 528,664건) | 10,243 | 49.8% | 74.9% / 79.3% | 0.5619 / 0.5626 |
| 시장 모델, 2026-10 순수 미래 (배포 모델과 같은 2026-09 이전 학습) | 12,544 | 49.9% | 75.4% / 79.8% | 0.5720 / 0.5725 |

홀드아웃 구간에서도 커버리지 50%·Q75 유효율 75% 안팎이 유지된다. 즉 "Q75 기준 투찰가가 약 75% 확률로 유효"라는 약속이 실제로 지켜진다.

**LLM 인터페이스**는 `bid_tools.py`의 함수(`predict_bid_simple`, `predict_bid_by_notice`, `predict_bid`, `evaluate_bid`, `compare_manual`, `backtest_summary`, `model_info`, `rule_basis`)를 두 경로로 노출한다.

- `ollama_chat.py`: Ollama 의 tool calling 모델이 자연어 질문을 해석해 도구를 호출하고 결과를 설명한다. 기본 `gemma4:26b`(약 50초, 정확). `gemma4:latest`(8B)는 30초로 빠르지만 인자 누락·되묻기가 잦다.
- `mcp_server.py`: 같은 도구를 MCP(stdio)로 노출한다. 프로젝트 `.mcp.json`에 등록되어 Claude Code 에서 바로 쓸 수 있다.

LLM 은 숫자를 만들지 않는다. 모든 예정가격·투찰가·확률은 `predict_cli`/`evaluate` 가 계산한 값이며 LLM 은 질문 해석과 설명만 맡는다.

공개 모델(시장 모델)은 허깅페이스 `casareborgia/yega-rate-predictor-market` 에 `publish_hf.py --profile public-market` 으로 올린다(사내 모델은 비공개 저장소). 기본은 dry-run 으로 올릴 파일 목록만 보여 주고, `--push` 를 붙여야 실제 업로드한다. 토큰은 로컬 `huggingface_hub` 설정(`~/.cache/huggingface/token`)만 쓴다.

예가 규정 변경 감시는 `monitor_regime.py` 로 한다. 지정한 달(기본 지난달) 1개월만 수집해(약 320호출) 범위별 사정률 평균·표준편차가 기준(±2%: 99.987/0.509, ±3%: 99.855/0.740)에서 평균 0.10%p 또는 표준편차 0.05 이상 벗어나는지와, 그 달 홀드아웃 백테스트의 커버리지(45~55%)·Q75 유효율(72~78%)을 확인해 `out/regime_check_<YYYY-MM>.md` 에 쓴다. 벗어나면 종료 코드 3 이다. 연 1회 실행을 권장한다.

계약 법령·예규·공공기관 규정 변경 감시는 `monitor_rules.py` 로 한다(Phase 8). `data/rules/registry.json` 에 등록된 국가·지방계약법 시행령, (계약예규) 예정가격작성기준, 지자체 입찰 및 계약 집행기준, 한국남동발전 계약규정의 일련번호·시행일을 조회해 변경 여부를 점검한다(종료 코드 0=변경없음, 3=모델관련변경, 4=API오류, 6=무관변경만).

## 입찰 시뮬레이터 (학습용, `src/sim/`)

하한 통과(Q75 ≈ 75%)와 낙찰은 다르다. 시뮬레이터는 "낙찰확률을 최대로 하는 투찰가"를 단계적으로 다룬다.

```bash
uv run python src/sim/validate_env.py                                   # 1단계: 예비가격 15개·4개 추첨 환경 vs 실측 사정률 분포 (out/sim_env_check.md)
uv run python src/sim/collect_bidders.py --month 2026-10 --max-calls 600 # 업체별 개찰 결과 수집(업체명 미저장, 사업자번호 해시)
uv run python src/sim/backtest_strategies.py --month 2026-10            # 2단계: 전략 재연 백테스트 (out/sim_strategy_backtest_*.md)
```

- 환경(`env.py`): 2026-10 예비가격상세 12,601건으로 확인한 규칙 — ±2%는 15등분 칸마다 1개, ±3%는 아래 8칸·위 7칸, 번호 선택은 가격과 무관하고 최다 선택 4개 평균이 예정가격. ±2% 분포는 실측과 일치(std 0.512 vs 0.508), ±3%는 추첨의 저가 쪽 치우침을 β로 보정(평균 99.854 vs 99.855, std 0.767 vs 0.740).
- 최적화(`optimize.py`): 투찰가를 "그 금액이 낙찰하한이 되는 사정률" r 로 바꾸면 1순위 조건은 "r ≥ 실제 사정률, 그 사이에 경쟁사 없음". 경쟁사를 독립 경험 분포로 보고 P(낙찰|r) 을 계산한다.
- 2026-10 적격심사 용역 599건 재연(앞 절반 적합 → 뒤 300건 평가): 낙찰률 Q75 5.7% / Q50 10.3% / Q25 12.7% / MC 최적 12.3% / 무작위 기준 7.0%. 경쟁사 환산 사정률은 실제 사정률보다 높게 몰려(중앙값 약 100.1) 낮게 쓰는 쪽이 유리하다. 경쟁사 30곳 이상에서는 독립 가정이 맞지 않아 MC 최적이 과도하게 공격적이다 → 다음 단계(컨텍스트·밴딧)에서 다룬다.

## 예측 CLI 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--bid_no` | 없음 | 나라장터 입찰공고번호. 입찰공고정보서비스로 추정가격·기초금액·예가 범위·낙찰하한율·A값(공사)·발주기관·공고일·용역구분을 채우고 시장 모델을 쓴다. 기초금액은 ×1.1 대신 공고값. 직접 준 인자가 공고값보다 우선 |
| `--est_price` | (필수, `--bid_no` 면 생략) | 추정가격, 부가세 제외 원 |
| `--work_type` | 2 | 1=공사, 2=용역 (`--bid_no` 면 공고값) |
| `--bidder_count` | 사내 8 / 시장 39 | 예상 참여업체수 (생략 시 각 모델의 중앙값) |
| `--lower_limit` | 규칙표 자동 | 낙찰하한율 %. 생략 시 규칙표(`data/rules/lower_limit_rules.json`)에서 판정 — 사내 모델은 한국남동발전(조달청 기준 준용), 시장 모델은 `--agency`. 판정 불가 시 87.745. 명시값 우선 |
| `--servc_kind` | 높은 쪽 | 용역 종류. general=일반용역(청소·경비·시설관리·학술연구), tech=기술용역(설계·감리·엔지니어링). 낙찰하한율 판정용 |
| `--a_value` | 0 (`--bid_no` 면 공고값) | A값(원, **공사만**). 공고문의 국민연금·건강보험·노인장기요양·퇴직공제부금·산업안전보건관리비 등 합계. 하한금액 = (예정가격 − A) × 하한율 + A. 통과확률은 그대로 |
| `--award_method` | 적격심사 | 적격심사 / 제한적최저가 / 최저가 / 협상 |
| `--dept` | 없음 | 발주부서명 (사내 모델만). 30건 미만 부서는 "기타" |
| `--agency` | 없음 | 발주기관명 (예: 서울특별시 강남구, 경기도교육청, 한국도로공사). 예가 범위·낙찰하한율 자동 판정 |
| `--as_of` | 오늘 | 기준일. 이 날짜 이전 이력으로 이동통계 계산 |
| `--margin` | 0.1 | 낙찰하한금액 위 안전마진 % |
| `--round_unit` | 1000 | 투찰금액 올림 단위 원 |
| `--basis` | Q75 | 권장 투찰가 기준 분위수. Q75 ≈ 통과확률 75%(안전), Q50 ≈ 50%, Q25 ≈ 25%(공격적) |
| `--market` | | 시장 모델(나라장터) 사용. 공공발주 전체용. `--range` 또는 `--agency` 필요 |
| `--range` | | 예가 범위 % (2 또는 3). 공고문 "기초금액 ±N% 범위 내 15개 예비가격"의 N. 시장 입찰의 86%가 3 |
| `--negotiated` | | 수의계약 모드 (사내 모델만). ML 대신 학습에서 분리한 수의계약 사정률의 경험 분포를 쓴다 |
| `--json` | | JSON 출력 (MCP 도구화 대비) |

권장 투찰금액은 기준 분위수(기본 Q75)의 낙찰하한금액에 안전마진을 더한 값이다.
**낙찰하한금액은 실제 예정가격에 비례하므로, Qα 기준 투찰가는 실제 사정률이 Qα 이하로 나올 때만 유효하다.**
실제 사정률이 기준보다 높게 나오면 하한선이 투찰가 위로 올라가 하한 미달(무효)이 된다. 따라서 통과확률은 Q75 약 75%, Q50 약 50%, Q25 약 25%이며,
낮은 분위수를 기준으로 할수록 가격은 유리하지만 무효 위험이 커진다. (2026-10-18 이전 버전은 이 방향이 반대로 표시되어 Q25를 "통과 75%"로 권장했다. 수정됨.)

## 현재 성능

### 시장 모델 (나라장터 538,907건, 2026-10-31)

| 지표 | 값 | 기준 |
|---|---|---|
| Baseline MAE (예가 범위별 평균) | 0.571%p | |
| ML MAE (검증) | 0.570%p (+0.11%) | 악화 없음 |
| Overfit Gap | -0.001%p | ≤ 0.08%p 통과 |
| Q25~Q75 커버리지 | 50.1% | 45~55% 통과 |
| 기준별 투찰가 통과율 | Q25 25.0% / Q50 50.0% / Q75 75.1% | 분위수 보정 적중 |

사정률 분포는 예가 범위가 결정한다. ±2%는 평균 99.99·표준편차 0.51, ±3%는 평균 99.86·표준편차 0.74이며, 공사/용역·참여업체수·기관·규모·월의 영향은 0.1%p 이내다.
따라서 `--range`를 정확히 넣거나 `--agency`로 검증된 발주기관(지자체·교육청·주요 공기업 등)을 지정하는 것이 가장 중요하다.

### 사내 모델

사내 경쟁입찰 데이터로 같은 방식(시계열 CV, 분위수 보정)을 적용했다. 결과 수치는 사내 정보라 공개하지 않는다.
경쟁입찰 사정률은 예가 15개 중 4개 추첨의 결과(±2% 이론 표준편차 0.51)라 MAE는 하한에 있고, ML이 Baseline과 같은 것이 정상이다.
이 모델의 가치는 오차 개선이 아니라 보정된 분위수 구간, 즉 "Q75 기준 투찰가가 실제로 약 75% 확률로 유효하다"는 통과확률의 정확성이다.
수의계약은 예가 추첨 절차가 아니므로 학습에서 분리하고 `--negotiated`로 경험 분포를 직접 쓴다.
나라장터 시장 데이터를 가중치·범위 필터 등 어떤 구성으로 합쳐도 사내 검증 MAE는 개선되지 않아 사내 단독 학습이 기본이다.

## 재학습

데이터가 갱신되면 Phase 1(또는 Phase 2 `--build`) 후 `train_model.py`를 다시 실행한다. 모델 파일에는 인코더·사전값·최근 15건 이력이 함께 저장되어 CLI가 단독으로 로드한다.

## 구조

```
src/preprocess_internal.py   사내 CSV 정제
src/collect_g2b_api.py       나라장터 API 수집기 (market / agency 모드)
src/features.py              피처 생성 (학습·추론 공용, shift(1) 규칙, 시장 피처 MARKET_FEATURES)
src/train_model.py           분위수 모델 학습 + 시계열 CV (--source internal|market)
src/predict_cli.py           예측 CLI
src/evaluate.py              실제값 대비 검증 (공고번호 단건, 홀드아웃 백테스트)
src/bid_tools.py             LLM 인터페이스용 도구 함수 (MCP·Ollama 공용)
src/mcp_server.py            MCP 서버 (.mcp.json)
src/ollama_chat.py           Ollama gemma4 자연어 인터페이스
src/rules.py                 발주기관 기반 예가 범위·낙찰하한율 규칙 엔진 (Phase 8)
src/law_client.py            법제처 Open API / korean-law-mcp 연결 클라이언트 (Phase 8)
src/monitor_rules.py         계약 법령·행정규칙·기관 규정 변경 감시 (Phase 8)
tests/                       pytest (오프라인 + 실데이터 불변식)
```

## 라이선스

MIT ([LICENSE](LICENSE)). 데이터 출처: 공공데이터포털 조달청 나라장터 Open API, 법제처 국가법령정보 Open API.
