"""허깅페이스 모델 저장소 업로드 스크립트.

프로필
  private       (기본) 사내·시장 모델과 전체 리포트 → 비공개 저장소 casareborgia/yega-rate-predictor
  public-market 시장 모델(공개 나라장터 데이터로만 학습)과 시장 리포트 → 공개 저장소 casareborgia/yega-rate-predictor-market.
                사내 계약 이력·부서명·수의계약 분포·사내 리포트가 섞이지 않았는지 업로드 전에 검사한다(check_public_safe).

사용 예
  # 업로드 대상 파일 매니페스트 확인 (dry-run, 기본값)
  uv run python src/publish_hf.py

  # 실제 비공개 저장소에 업로드
  uv run python src/publish_hf.py --push

  # 커밋 메시지 직접 지정 및 다른 저장소 지정
  uv run python src/publish_hf.py --push --repo username/custom-repo --message "모델 업데이트"

  # 공개용(시장 모델만) 매니페스트 확인 / 업로드
  uv run python src/publish_hf.py --profile public-market
  uv run python src/publish_hf.py --profile public-market --push
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import joblib

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_REPO = "casareborgia/yega-rate-predictor"
MODEL_FILES = ["bidding_model.joblib", "market_model.joblib"]          # models/ 에 둘 다 있어야 함(없으면 오류)
REPORT_REQUIRED = ["cv_report.md", "cv_report_market.md"]              # out/ 에 있어야 함(없으면 오류)
REPORT_OPTIONAL_GLOBS = ["cv_compare.md", "backtest_*.md"]             # 있으면 포함
CARD_PATH = ROOT / "docs" / "hf_model_card.md"                         # README.md 로 업로드
PROFILES = {
    "private": {"repo": DEFAULT_REPO, "models": MODEL_FILES, "required": REPORT_REQUIRED,
                "optional": REPORT_OPTIONAL_GLOBS, "card": CARD_PATH, "private": True},
    "public-market": {"repo": "casareborgia/yega-rate-predictor-market", "models": ["market_model.joblib"],
                      "required": ["cv_report_market.md"], "optional": ["backtest_market_*.md"],
                      "card": ROOT / "docs" / "hf_model_card_public.md", "private": False},
}
# 공개 업로드 금지 표식: 사내 계약번호(C006…), 사내 데이터 언급
PUBLIC_FORBIDDEN_TEXT = ("C006", "사내", "계약대장", "남동발전")


def collect_files(models_dir: Path, out_dir: Path, card_path: Path, profile: str = "private") -> list[tuple[Path, str]]:
    """업로드 대상 파일 목록을 수집한다.

    반환: [(로컬 경로, 저장소 내 경로), ...]
    저장소 경로:
      - 모델: models/<이름>
      - 리포트: reports/<이름> (파일명의 '~'는 '_'로 치환)
      - 카드: README.md
    """
    prof = PROFILES[profile]
    missing: list[str] = []
    for m in prof["models"]:
        p = models_dir / m
        if not p.is_file():
            missing.append(f"{m} ({p})")

    for r in prof["required"]:
        p = out_dir / r
        if not p.is_file():
            missing.append(f"{r} ({p})")

    if not card_path.is_file():
        missing.append(f"{card_path.name} ({card_path})")

    if missing:
        raise FileNotFoundError(f"필수 파일이 없습니다: {', '.join(missing)}")

    files: list[tuple[Path, str]] = []

    # 모델 카드 -> 저장소 루트 README.md
    files.append((card_path, "README.md"))

    # 모델 아티팩트 2종
    for m in prof["models"]:
        files.append((models_dir / m, f"models/{m}"))

    # 필수 리포트
    for r in prof["required"]:
        dest_name = r.replace("~", "_")
        files.append((out_dir / r, f"reports/{dest_name}"))

    # 선택 리포트 (이름순 정렬)
    optional_found: dict[str, Path] = {}
    for pattern in prof["optional"]:
        for p in out_dir.glob(pattern):
            if p.is_file() and p.name not in prof["required"]:
                optional_found[p.name] = p

    for name in sorted(optional_found.keys()):
        p = optional_found[name]
        dest_name = name.replace("~", "_")
        files.append((p, f"reports/{dest_name}"))

    return files


def check_public_safe(files: list[tuple[Path, str]]) -> None:
    """공개 업로드 전 검사: 모델은 시장 모델이어야 하고 사내 이력·수의계약 분포·사내 부서명이 없어야 하며,
    텍스트 파일(카드·리포트)에 사내 계약번호·사내 데이터 언급이 없어야 한다. 위반 시 ValueError."""
    problems = []
    for local_p, repo_p in files:
        if repo_p.startswith("models/"):
            art = joblib.load(local_p)
            depts = set((art.get("encoders") or {}).get("dept_code", []))
            if art.get("source") != "market":
                problems.append(f"{repo_p}: 시장 모델이 아님(source={art.get('source')})")
            if art.get("history_tail") is not None or art.get("negotiated_dist"):
                problems.append(f"{repo_p}: 사내 이력(history_tail) 또는 수의계약 분포가 들어 있음")
            if depts - {"시장", "기타"}:
                problems.append(f"{repo_p}: 사내 부서명 인코더 {sorted(depts - {'시장', '기타'})}")
        else:
            text = local_p.read_text(encoding="utf-8", errors="ignore")
            hits = [w for w in PUBLIC_FORBIDDEN_TEXT if w in text]
            if hits:
                problems.append(f"{repo_p}: 공개 금지 표식 {hits}")
    if problems:
        raise ValueError("공개 업로드 검사 실패:\n  " + "\n  ".join(problems))


def read_model_meta(path: Path) -> dict[str, Any]:
    """joblib 아티팩트에서 메타데이터 4종을 읽어 반환한다."""
    artifact = joblib.load(path)
    keys = ["source", "trained_at", "n_train", "data_range"]
    if isinstance(artifact, dict):
        return {k: artifact.get(k) for k in keys}
    return {k: getattr(artifact, k, None) for k in keys}


def stage(files: list[tuple[Path, str]], dest: Path) -> Path:
    """dest 아래에 저장소 디렉터리 구조대로 파일을 복사한다."""
    dest.mkdir(parents=True, exist_ok=True)
    for src, repo_path in files:
        target = dest / repo_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    return dest


def build_manifest(files: list[tuple[Path, str]], metas: dict[str, dict[str, Any]]) -> str:
    """업로드할 파일 목록과 크기, 메타데이터 표를 생성한다."""
    lines = [
        "| 저장소 경로 | 크기(바이트) | 비고 |",
        "|---|---|---|",
    ]
    total_size = 0
    for local_path, repo_path in files:
        size = local_path.stat().st_size if local_path.exists() else 0
        total_size += size
        note = ""
        meta = metas.get(repo_path) or metas.get(local_path.name)
        if meta:
            src = meta.get("source") or "internal"
            t = meta.get("trained_at") or ""
            n = f"{meta['n_train']:,}건" if meta.get("n_train") is not None else ""
            parts = [p for p in [src, t, n] if p]
            note = " / ".join(parts)
        lines.append(f"| {repo_path} | {size:,} | {note} |")

    lines.append(f"| **총 {len(files)}개 파일** | **{total_size:,}** | |")
    return "\n".join(lines)


def default_commit_message(metas: dict[str, dict[str, Any]] | list[dict[str, Any]]) -> str:
    """모델 메타데이터를 기반으로 기본 커밋 메시지를 생성한다."""
    if isinstance(metas, dict):
        seen: set[tuple[Any, Any, Any]] = set()
        model_metas: list[dict[str, Any]] = []
        for v in metas.values():
            if isinstance(v, dict):
                key = (v.get("source"), v.get("trained_at"), v.get("n_train"))
                if key not in seen:
                    seen.add(key)
                    model_metas.append(v)
    elif isinstance(metas, (list, tuple)):
        model_metas = list(metas)
    else:
        model_metas = []

    def _sort_key(m: dict[str, Any]) -> tuple[int, str]:
        src = str(m.get("source") or "internal")
        return (0 if src == "internal" else 1, src)

    model_metas.sort(key=_sort_key)
    parts: list[str] = []
    for m in model_metas:
        src = m.get("source") or "internal"
        n = f"{m['n_train']:,}건" if m.get("n_train") is not None else "?건"
        t = f"({m['trained_at']})" if m.get("trained_at") else ""
        parts.append(f"{src} {n}{t}")

    summary = " / ".join(parts)
    if summary:
        return f"모델 갱신: {summary} + 리포트"
    return "모델 갱신 + 리포트"


def publish(
    repo_id: str,
    staged_dir: Path,
    message: str,
    *,
    api: Any = None,
    allow_public: bool = False,
    private: bool = True,
) -> dict[str, Any]:
    """스테이징된 디렉터리를 허깅페이스 저장소에 업로드한다. private=False(공개 프로필)면 공개 저장소로 만든다."""
    if api is None:
        from huggingface_hub import HfApi  # lazy import
        api = HfApi()

    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    info = api.repo_info(repo_id, repo_type="model")
    if private and not getattr(info, "private", False) and not allow_public:
        raise RuntimeError(
            f"저장소가 공개 상태입니다: {repo_id}. 공개 저장소 업로드를 허용하려면 --allow-public 옵션을 사용하세요."
        )

    res = api.upload_folder(
        folder_path=str(staged_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=message,
    )
    commit_url = getattr(res, "commit_url", None)
    n_files = sum(1 for p in staged_dir.rglob("*") if p.is_file())
    return {
        "repo_id": repo_id,
        "commit_url": commit_url,
        "n_files": n_files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="허깅페이스 모델 및 리포트 업로드")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="private",
                        help="private=사내+시장 비공개(기본), public-market=시장 모델만 공개 저장소")
    parser.add_argument("--repo", default=None, help=f"저장소 ID (기본: 프로필별, private={DEFAULT_REPO})")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models", help="모델 디렉터리 경로")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out", help="리포트 디렉터리 경로")
    parser.add_argument("--card", type=Path, default=None, help="모델 카드 경로 (기본: 프로필별)")
    parser.add_argument("--push", action="store_true", help="실제 업로드 수행 (기본은 dry-run)")
    parser.add_argument("--message", default=None, help="커밋 메시지 (기본: 자동 생성)")
    parser.add_argument("--allow-public", action="store_true", help="공개 저장소 업로드 허용")

    args = parser.parse_args(argv)
    prof = PROFILES[args.profile]
    args.repo = args.repo or prof["repo"]
    args.card = args.card or prof["card"]

    try:
        files = collect_files(args.models_dir, args.out_dir, args.card, profile=args.profile)
        if not prof["private"]:
            check_public_safe(files)
    except (FileNotFoundError, ValueError) as e:
        sys.stderr.write(f"오류: {e}\n")
        return 1

    metas: dict[str, dict[str, Any]] = {}
    for local_p, repo_p in files:
        if repo_p.startswith("models/"):
            meta = read_model_meta(local_p)
            metas[repo_p] = meta
            metas[local_p.name] = meta

    manifest = build_manifest(files, metas)
    print(manifest)

    if not args.push:
        print("\n`--push` 를 붙이면 업로드합니다.")
        return 0

    message = args.message if args.message else default_commit_message(metas)

    with tempfile.TemporaryDirectory() as tmpdir:
        staged_dir = stage(files, Path(tmpdir))
        try:
            res = publish(
                args.repo,
                staged_dir,
                message,
                allow_public=args.allow_public,
                private=prof["private"],
            )
        except RuntimeError as e:
            sys.stderr.write(f"오류: {e}\n")
            return 2

    res_dict = res or {"repo_id": args.repo, "commit_url": None, "n_files": len(files)}
    print(f"\n업로드 완료: {res_dict['repo_id']} ({res_dict['n_files']}개 파일)")
    if res_dict.get("commit_url"):
        print(f"커밋 URL: {res_dict['commit_url']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
