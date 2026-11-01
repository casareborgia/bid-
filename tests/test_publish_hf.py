"""publish_hf.py 단위 테스트.

네트워크·토큰·실데이터 없이 가짜 아티팩트와 가짜 HfApi를 사용하여 전체 파이프라인을 검증한다.
"""
from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pytest

import publish_hf as PH


@pytest.fixture
def fake_env(tmp_path: Path) -> dict:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    card_path = tmp_path / "docs" / "hf_model_card.md"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text("# Model Card Content", encoding="utf-8")

    internal_meta = {
        "source": "internal",
        "trained_at": "2026-01-01T00:00:00",
        "n_train": 1000,
        "data_range": ["2020-01-01", "2025-12-31"],
    }
    market_meta = {
        "source": "market",
        "trained_at": "2026-01-02T00:00:00",
        "n_train": 20000,
        "data_range": ["2023-01-01", "2025-12-31"],
    }
    joblib.dump(internal_meta, models_dir / "bidding_model.joblib")
    joblib.dump(market_meta, models_dir / "market_model.joblib")

    # 필수 리포트
    (out_dir / "cv_report.md").write_text("# CV Report", encoding="utf-8")
    (out_dir / "cv_report_market.md").write_text("# Market CV Report", encoding="utf-8")

    # 선택 리포트 + 제외 대상 파일
    (out_dir / "cv_compare.md").write_text("# CV Compare", encoding="utf-8")
    (out_dir / "backtest_x_2026-01~2026-02.md").write_text("# Backtest", encoding="utf-8")
    (out_dir / "other.txt").write_text("ignore me", encoding="utf-8")

    return {
        "models_dir": models_dir,
        "out_dir": out_dir,
        "card_path": card_path,
        "internal_meta": internal_meta,
        "market_meta": market_meta,
    }


class FakeRepoInfo:
    def __init__(self, private: bool = True):
        self.private = private


class FakeCommitResult:
    def __init__(self, commit_url: str = "https://x"):
        self.commit_url = commit_url


class FakeApi:
    def __init__(self, private: bool = True):
        self.calls: list[tuple] = []
        self.private = private

    def create_repo(self, repo_id: str, repo_type: str = "model", private: bool = True, exist_ok: bool = True):
        self.calls.append(("create_repo", repo_id, repo_type, private, exist_ok))

    def repo_info(self, repo_id: str, repo_type: str = "model"):
        self.calls.append(("repo_info", repo_id, repo_type))
        return FakeRepoInfo(self.private)

    def upload_folder(self, folder_path: str, repo_id: str, repo_type: str = "model", commit_message: str = ""):
        self.calls.append(("upload_folder", folder_path, repo_id, repo_type, commit_message))
        return FakeCommitResult("https://x")


# 1. collect_files 경로 검증, ~ -> _ 치환, other.txt 제외, 선택 리포트 정렬
def test_collect_files_structure_and_sorting(fake_env: dict):
    files = PH.collect_files(fake_env["models_dir"], fake_env["out_dir"], fake_env["card_path"])
    repo_paths = [repo_p for _, repo_p in files]

    assert "README.md" in repo_paths
    assert "models/bidding_model.joblib" in repo_paths
    assert "models/market_model.joblib" in repo_paths
    assert "reports/cv_report.md" in repo_paths
    assert "reports/cv_report_market.md" in repo_paths
    # ~ -> _ 치환 확인
    assert "reports/backtest_x_2026-01_2026-02.md" in repo_paths
    assert not any("~" in p for p in repo_paths)
    # other.txt 제외 확인
    assert not any("other.txt" in p for p in repo_paths)

    # 선택 리포트 정렬 확인
    reports = [p for p in repo_paths if p.startswith("reports/")]
    # 필수 리포트 외 선택 리포트 추출
    optionals = [p for p in reports if p not in ("reports/cv_report.md", "reports/cv_report_market.md")]
    assert optionals == sorted(optionals)


# 2. 필수 파일 누락 시 FileNotFoundError 및 메시지에 파일명 모두 포함
def test_missing_required_files_reports_all_missing(fake_env: dict):
    (fake_env["models_dir"] / "bidding_model.joblib").unlink()
    (fake_env["out_dir"] / "cv_report.md").unlink()

    with pytest.raises(FileNotFoundError) as exc_info:
        PH.collect_files(fake_env["models_dir"], fake_env["out_dir"], fake_env["card_path"])

    msg = str(exc_info.value)
    assert "bidding_model.joblib" in msg
    assert "cv_report.md" in msg


# 3. read_model_meta 4개 키 반환 및 default_commit_message 천 단위 구분
def test_read_model_meta_and_commit_message(fake_env: dict):
    meta = PH.read_model_meta(fake_env["models_dir"] / "bidding_model.joblib")
    assert set(meta.keys()) == {"source", "trained_at", "n_train", "data_range"}
    assert meta["source"] == "internal"
    assert meta["trained_at"] == "2026-01-01T00:00:00"
    assert meta["n_train"] == 1000
    assert meta["data_range"] == ["2020-01-01", "2025-12-31"]

    market_meta = PH.read_model_meta(fake_env["models_dir"] / "market_model.joblib")
    msg = PH.default_commit_message({
        "models/bidding_model.joblib": meta,
        "models/market_model.joblib": market_meta,
    })
    assert "1,000건" in msg
    assert "20,000건" in msg
    assert "internal" in msg
    assert "market" in msg
    assert "+ 리포트" in msg


# 4. stage 구조 복사 및 build_manifest 검증
def test_stage_and_build_manifest(fake_env: dict, tmp_path: Path):
    files = PH.collect_files(fake_env["models_dir"], fake_env["out_dir"], fake_env["card_path"])
    metas = {
        "models/bidding_model.joblib": fake_env["internal_meta"],
        "models/market_model.joblib": fake_env["market_meta"],
    }
    dest = tmp_path / "staged"
    staged = PH.stage(files, dest)
    assert staged == dest

    for _, repo_p in files:
        assert (dest / repo_p).is_file()

    manifest = PH.build_manifest(files, metas)
    for _, repo_p in files:
        assert repo_p in manifest
    assert "총" in manifest


# 5. publish 가짜 api 주입 시 create_repo, upload_folder 호출 인자 및 반환값 검증
def test_publish_success(tmp_path: Path):
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    (staged_dir / "file1.txt").write_text("sample")

    api = FakeApi(private=True)
    res = PH.publish("test/repo", staged_dir, "test commit", api=api)

    assert ("create_repo", "test/repo", "model", True, True) in api.calls
    upload_calls = [c for c in api.calls if c[0] == "upload_folder"]
    assert len(upload_calls) == 1
    assert upload_calls[0][1] == str(staged_dir)
    assert upload_calls[0][2] == "test/repo"
    assert upload_calls[0][3] == "model"
    assert upload_calls[0][4] == "test commit"

    assert res == {
        "repo_id": "test/repo",
        "commit_url": "https://x",
        "n_files": 1,
    }


# 6. repo_info private=False 시 RuntimeError 및 allow_public=True 처리
def test_publish_public_repo_safety(tmp_path: Path):
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    (staged_dir / "file1.txt").write_text("sample")

    api = FakeApi(private=False)
    with pytest.raises(RuntimeError) as exc_info:
        PH.publish("test/repo", staged_dir, "test commit", api=api, allow_public=False)

    assert "공개" in str(exc_info.value)
    assert not any(c[0] == "upload_folder" for c in api.calls)

    api2 = FakeApi(private=False)
    res = PH.publish("test/repo", staged_dir, "test commit", api=api2, allow_public=True)
    assert any(c[0] == "upload_folder" for c in api2.calls)
    assert res["repo_id"] == "test/repo"


# 7. main dry-run 이 0 반환, README.md 와 --push 안내 출력, huggingface_hub 미임포트 확인
def test_main_dry_run(fake_env: dict, capsys: pytest.CaptureFixture):
    if "huggingface_hub" in sys.modules:
        pytest.skip("huggingface_hub already imported in this session")

    ret = PH.main([
        "--models-dir", str(fake_env["models_dir"]),
        "--out-dir", str(fake_env["out_dir"]),
        "--card", str(fake_env["card_path"]),
    ])
    assert ret == 0
    captured = capsys.readouterr()
    assert "README.md" in captured.out
    assert "--push" in captured.out
    assert "huggingface_hub" not in sys.modules


# 8. main --push 호출 시 monkeypatch 로 publish 인자 검증
def test_main_push_with_monkeypatch(fake_env: dict, monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []

    def fake_publish(repo_id: str, staged_dir: Path, message: str, *, api=None, allow_public: bool = False,
                     private: bool = True):
        calls.append({
            "repo_id": repo_id,
            "staged_dir": staged_dir,
            "message": message,
            "allow_public": allow_public,
            "private": private,
            "files": sorted(str(p.relative_to(staged_dir)) for p in staged_dir.rglob("*") if p.is_file()),
        })
        return {"repo_id": repo_id, "commit_url": "https://fake/commit", "n_files": 5}

    monkeypatch.setattr(PH, "publish", fake_publish)

    ret = PH.main([
        "--models-dir", str(fake_env["models_dir"]),
        "--out-dir", str(fake_env["out_dir"]),
        "--card", str(fake_env["card_path"]),
        "--push",
        "--repo", "custom/repo",
        "--message", "커스텀 커밋 메시지",
        "--allow-public",
    ])
    assert ret == 0
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "custom/repo"
    assert calls[0]["message"] == "커스텀 커밋 메시지"
    assert calls[0]["allow_public"] is True

    # 기본 커밋 메시지 생성 검증
    calls.clear()
    ret = PH.main([
        "--models-dir", str(fake_env["models_dir"]),
        "--out-dir", str(fake_env["out_dir"]),
        "--card", str(fake_env["card_path"]),
        "--push",
    ])
    assert ret == 0
    assert len(calls) == 1
    assert calls[0]["repo_id"] == PH.DEFAULT_REPO
    assert "모델 갱신:" in calls[0]["message"]
    assert calls[0]["allow_public"] is False


# 9. main 예외 처리: 필수 파일 누락 시 1 반환, 공개 저장소 오류 시 2 반환
def test_main_error_exit_codes(fake_env: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    (fake_env["models_dir"] / "bidding_model.joblib").unlink()
    ret = PH.main([
        "--models-dir", str(fake_env["models_dir"]),
        "--out-dir", str(fake_env["out_dir"]),
        "--card", str(fake_env["card_path"]),
    ])
    assert ret == 1
    captured = capsys.readouterr()
    assert "오류:" in captured.err

    # 복구 후 공개 저장소 에러 모의
    joblib.dump(fake_env["internal_meta"], fake_env["models_dir"] / "bidding_model.joblib")

    def fake_publish_err(*args, **kwargs):
        raise RuntimeError("저장소가 공개 상태입니다")

    monkeypatch.setattr(PH, "publish", fake_publish_err)
    ret2 = PH.main([
        "--models-dir", str(fake_env["models_dir"]),
        "--out-dir", str(fake_env["out_dir"]),
        "--card", str(fake_env["card_path"]),
        "--push",
    ])
    assert ret2 == 2
    captured2 = capsys.readouterr()
    assert "오류:" in captured2.err


# 10. 공개 프로필: 시장 모델·시장 리포트만, 사내 흔적이 있으면 거부
def test_public_market_profile(fake_env: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    calls: list[dict] = []
    def fake_publish(repo_id, staged_dir, message, *, api=None, allow_public=False, private=True):
        calls.append({"repo_id": repo_id, "private": private,
                      "files": sorted(str(p.relative_to(staged_dir)) for p in staged_dir.rglob("*") if p.is_file())})
        return {"repo_id": repo_id, "commit_url": None, "n_files": len(calls[-1]["files"])}
    monkeypatch.setattr(PH, "publish", fake_publish)
    (fake_env["out_dir"] / "backtest_market_2026-10.md").write_text("# 시장 백테스트", encoding="utf-8")
    (fake_env["out_dir"] / "backtest_internal_2026-01~2026-10.md").write_text("| C0062620004 |", encoding="utf-8")
    common = ["--profile", "public-market", "--models-dir", str(fake_env["models_dir"]),
              "--out-dir", str(fake_env["out_dir"]), "--card", str(fake_env["card_path"])]
    assert PH.main(common + ["--push"]) == 0
    c = calls[0]
    assert c["repo_id"] == PH.PROFILES["public-market"]["repo"] and c["private"] is False
    assert c["files"] == ["README.md", "models/market_model.joblib", "reports/backtest_market_2026-10.md",
                          "reports/cv_report_market.md"]

    # 카드에 사내 언급이 있으면 업로드 전에 거부(1)
    fake_env["card_path"].write_text("사내 계약대장 기반", encoding="utf-8")
    assert PH.main(common + ["--push"]) == 1 and len(calls) == 1
    assert "공개 업로드 검사 실패" in capsys.readouterr().err


def test_check_public_safe_rejects_internal_artifact(tmp_path: Path):
    art = tmp_path / "m.joblib"
    joblib.dump({"source": "internal", "history_tail": [1], "encoders": {"dept_code": ["영흥발전본부", "기타"]}}, art)
    with pytest.raises(ValueError) as e:
        PH.check_public_safe([(art, "models/market_model.joblib")])
    msg = str(e.value)
    assert "시장 모델이 아님" in msg and "history_tail" in msg and "영흥발전본부" in msg
