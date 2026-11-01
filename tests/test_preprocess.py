"""Phase 1 정제 스크립트 테스트. 원본 CSV가 없으면 실데이터 테스트는 건너뛴다."""
import numpy as np
import pandas as pd
import pytest

import preprocess_internal as pi


# ---------- 단위 테스트 (원본 파일 불필요) ----------

def test_parse_number():
    assert pi.parse_number("14,661,820,800") == 14661820800.0
    assert pi.parse_number(" 80.52 ") == 80.52
    assert np.isnan(pi.parse_number(""))
    assert np.isnan(pi.parse_number("-"))


def test_parse_duration_extracts_digits_and_clips():
    assert pi.parse_duration("730일") == 730
    assert pi.parse_duration("99999일") == 3650
    assert np.isnan(pi.parse_duration(""))


def _row(contract_no, date, wt, base, design, yega, method="제한경쟁", bidders="3"):
    r = [""] * pi.EXPECTED_COLS
    r[0] = "테스트본부"; r[1] = contract_no; r[2] = date; r[3] = wt; r[4] = method; r[5] = "적격심사"
    r[14] = design; r[16] = design; r[18] = base; r[19] = yega; r[39] = "365일"; r[52] = bidders
    return r


def test_clean_filters_by_rules():
    rows = [
        _row("C0062110001", "2021/01/05", "공사", "1,000,000", "1,000,000", "998,500"),   # 정상 (99.85)
        _row("C0062120002", "2021/02/05", "용역", "1,000,000", "1,000,000", "1,050,000"), # 범위 밖 (105)
        _row("C0062110003", "2018/03/05", "공사", "1,000,000", "1,000,000", "999,000"),   # 2019 미만
        _row("C0062110004", "2022/03/05", "공사", "900,000", "1,000,000", "899,000"),     # 기초≠설계
        _row("C0062110005", "2022/04/05", "공사", "0", "1,000,000", "0", method="수의"),   # 수의 (기초 0)
        _row("C0062120006", "2022/12/05", "용역", "2,000,000", "2,000,000", "2,004,000", bidders=""),  # 정상, 결측 업체수
        _row("C0062120007", "2023/01/05", "용역", "3,000,000", "3,000,000", "2,940,000", method="수의", bidders="1"),  # 수의 (기초 있음) → 정제본에 남고 학습 시 분리
    ]
    raw = pd.DataFrame(rows)
    df = pd.DataFrame({name: raw[idx] for idx, name in pi.COLUMN_MAP.items()})
    out, excluded, stats = pi.clean(df)

    assert stats["rows_clean"] == 3
    assert set(out["contract_no"]) == {"C0062110001", "C0062120006", "C0062120007"}
    assert stats["rows_negotiated_kept"] == 1                                   # 기초금액 있는 수의계약은 정제본에 남는다
    assert stats["rows_ratio_mismatch"] == 1 and excluded.iloc[0]["contract_no"] == "C0062110004"
    assert stats["rows_rate_out_of_range"] == 1
    assert (out["year"] >= 2019).all()
    assert out.loc[out["contract_no"] == "C0062120006", "bidder_count"].item() == 1  # 결측 → 1
    assert out.loc[out["contract_no"] == "C0062120006", "is_year_end"].item() == 1
    assert list(out["work_type"]) == [1, 2, 2]  # 공사=1, 용역=2, 날짜순 정렬
    assert stats["work_type_mismatch_vs_contract_no"] == 0
    assert (out["source"] == "internal").all()


# ---------- 실데이터 테스트 ----------

needs_raw = pytest.mark.skipif(not pi.DEFAULT_INPUT.exists(), reason="원본 CSV 없음")


@pytest.fixture(scope="module")
def real_clean():
    df = pi.load_raw(pi.DEFAULT_INPUT)
    return pi.clean(df)


@needs_raw
def test_real_row_counts(real_clean):
    out, _, stats = real_clean
    assert stats["raw_rows"] == 2593
    assert 1700 <= stats["rows_clean"] <= 2000
    assert len(out) == stats["rows_clean"]


@needs_raw
def test_real_rate_distribution(real_clean):
    out, _, stats = real_clean
    y = stats["yega_rate"]
    assert 99.70 <= y["mean"] <= 99.95
    assert 0.4 <= y["std"] <= 0.8
    assert out["yega_rate"].between(pi.RATE_MIN, pi.RATE_MAX).all()


@needs_raw
def test_real_invariants(real_clean):
    out, _, stats = real_clean
    assert out["year"].min() >= 2019
    assert (out["base_amt"] > 0).all()
    assert ((out["base_amt"] / out["design_amt"] - 1).abs() < pi.RATIO_TOL).all()
    assert stats["work_type_mismatch_vs_contract_no"] == 0
    assert stats["rows_negotiated_kept"] > 0             # 기초금액이 있는 수의계약: 필터를 통과하고 학습 시 분리
    assert list(stats["branch_codes"]) == ["C006"]      # branch_code는 상수 → 피처 금지 근거
    assert out["contract_date"].is_monotonic_increasing
    assert out["bidder_count"].notna().all()
