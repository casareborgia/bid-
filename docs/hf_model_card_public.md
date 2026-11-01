---
language:
  - ko
license: mit
library_name: lightgbm
pipeline_tag: tabular-regression
tags:
  - lightgbm
  - quantile-regression
  - tabular-regression
  - public-procurement
  - korea
  - bidding
  - 예정가격
  - 사정률
  - 나라장터
---

# 나라장터 예정가격(사정률) 분위수 예측 모델 — Korean Public Procurement Reserve-Price Quantile Model

한국 공공입찰(나라장터 복수예가)에서 **예정가격 / 기초금액 × 100 = 사정률**의 Q25 / Q50 / Q75 를 예측하는 LightGBM Quantile 회귀 모델입니다.
그 분위수로 낙찰하한선을 통과할 확률이 보정된 투찰가를 계산할 수 있습니다. 개인 학습용 프로젝트의 결과물입니다.

| 파일 | 학습 데이터 |
|---|---|
| `models/market_model.joblib` | 조달청 나라장터 낙찰정보서비스(공공데이터포털 Open API)의 복수예가 개찰 538,907건 (2024-01~2026-09 33개월 전수; 예가 범위 ±2% 77,688 / ±3% 461,219) |

공개된 나라장터 데이터로만 학습했습니다. 코드(수집·학습·추론 CLI·검증 도구)는 GitHub `casareborgia/bid-` 저장소에 있습니다.

## 핵심 결론

- **사정률은 추첨 결과입니다.** 기초금액 ±N% 범위의 예비가격 15개 중 입찰자 추첨으로 뽑힌 4개의 평균이 예정가격이라, 어떤 피처로도 설명되지 않는 노이즈가 대부분입니다. MAE 는 하한에 있고 여러 알고리즘이 표본오차 안에서 같습니다.
- 따라서 이 모델의 가치는 **MAE 가 아니라 보정된 분위수**입니다.
- 사정률 분포를 결정하는 것은 **예가 범위 하나**입니다. ±2%: 평균 99.99, std 0.51 / ±3%: 평균 99.86, std 0.74. 공사/용역·참여업체수·기관 유형·규모·월의 영향은 평균 0.1%p 이내입니다.
- **투찰가 통과확률의 방향**: 낙찰하한금액은 실제 예정가격에 비례하므로, Qα 기준 투찰가는 실제 사정률이 Qα **이하**일 때만 유효합니다(통과확률 ≈ α). Q75 안전(75%), Q50 50%, Q25 공격적(25%).
- **하한 통과 ≠ 낙찰**: 2026-10 적격심사 용역 874건의 1순위 투찰가는 실제 낙찰하한 대비 중앙값 +0.074% 위였습니다. Q75 투찰가는 낙찰확률을 최대로 하는 가격이 아닙니다.

## 성능

### 5-Fold 시계열 교차검증 (TimeSeriesSplit)

| n | Baseline MAE | ML MAE | Overfit Gap | Q25~Q75 커버리지 | 투찰가 유효율 Q25 / Q50 / Q75 |
|---|---|---|---|---|---|
| 538,907 | 0.5708 | 0.5702 (+0.1%) | -0.001 | 50.1% | 25.0 / 50.0 / 75.1% |

Baseline 은 예가 범위별 학습평균. 단위 %p. 상세: `reports/cv_report_market.md`.

### 홀드아웃 백테스트 (컷오프 이전 데이터로 재적합 → 미래 기간 실제값 비교)

| 홀드아웃 | 학습 구간 | n | 커버리지 | 유효율 Q25 / Q50 / Q75 (마진 0) | Q75 유효율 (안전마진 0.1%) |
|---|---|---|---|---|---|
| 2026-09 | ~2026-08 (528,664건) | 10,243 | 49.8% | 25.0 / 49.7 / 74.9% | 79.3% |
| 2026-10 (순수 미래) | ~2026-09 (배포 모델과 같은 구간) | 12,544 | 49.9% | 25.6 / 50.9 / 75.4% | 79.8% |

상세: `reports/backtest_market_2026-09.md`, `reports/backtest_market_2026-10.md`.

## 모델 사양

- LightGBM `objective=quantile`, α = 0.25 / 0.50 / 0.75 각 1개. `max_depth=3, num_leaves=7, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, learning_rate=0.03, n_estimators≤300`(조기종료: 학습 구간 끝 15% 홀드아웃), `reg_lambda=1.0, random_state=42`.
- 최종 예측 = 0.7 × ML + 0.3 × 사전값(예가 범위별 학습평균·경험 분위수). 분위수 교차 방지(Q25 ≤ Q50 ≤ Q75).
- 피처: `work_type`(공사1/용역2), `range_cls`(예가 범위 ±2/±3), `est_price_log`(log10 추정가격), `bidder_count`(결측은 중앙값), `is_year_end`.

## 아티팩트 구조 (joblib dict)

`models`(α→LGBMRegressor), `features`, `cat_features`, `encoders`, `priors_by_range`(±2/±3 별 사전값), `range_stats`, `range_classes`, `bidder_median`,
`train_mean`, `train_std`, `shrink_ml`(0.7), `n_estimators`, `trained_at`, `n_train`, `data_range`, `cv_summary`, `lgb_params`, `source`("market").

## 사용 예 (Python 3.12, lightgbm 4.7, pandas ≥ 3.0, scikit-learn ≥ 1.9)

```python
import joblib, numpy as np, pandas as pd

art = joblib.load("models/market_model.joblib")
row = pd.DataFrame([{
    "work_type": 1,                                           # 1=공사, 2=용역
    "range_cls": "±3",                                        # 공고문 "기초금액 ±N% 범위 내 15개 예비가격"의 N
    "est_price_log": np.log10(1_000_000_000),                 # 추정가격 10억
    "bidder_count": 30.0, "is_year_end": 0,
}])
row["work_type"] = pd.Categorical(row["work_type"], categories=[1, 2])
row["range_cls"] = pd.Categorical(row["range_cls"], categories=art["range_classes"])
pri = art["priors_by_range"]["±3"]
q = {a: art["shrink_ml"] * m.predict(row[art["features"]])[0] + (1 - art["shrink_ml"]) * pri[a] for a, m in art["models"].items()}
base_amt = round(1_000_000_000 * 1.1)                          # 기초금액: 공고값을 쓴다(예시는 추정가격 × 1.1)
lower = base_amt * q[0.75] / 100 * 0.87745                     # Q75 기준 낙찰하한(하한율은 공고값. 예시는 87.745%)
print(q, "Q75 기준 투찰가(하한 통과확률 약 75%):", round(lower * 1.001))
```

공사는 A값(국민연금·건강보험·퇴직공제부금 등 사후정산 항목)을 하한율에서 빼고 더합니다: 하한 = (예정가격 − A) × 하한율 + A.

## 한계와 주의

- 예가 범위(`range_cls`)를 잘못 넣으면 분위수가 크게 틀어집니다. 공고문을 확인하세요(시장 입찰의 86%가 ±3%).
- 시장 데이터의 11~12월 학습분이 적어 `is_year_end` 효과는 사실상 0입니다.
- 예측은 참고용이며 실제 입찰 결과를 보장하지 않습니다. 낙찰하한율·A값·예가 범위는 반드시 공고문 값을 쓰세요.
- 데이터 출처: 공공데이터포털 「조달청_나라장터 낙찰정보서비스」 Open API.

## 변경 이력

- 2026-10-31 538,907건 전수 학습.
- 2026-11-01 2026-10 순수 미래 검증 추가, 공개 저장소로 분리(시장 모델만).
