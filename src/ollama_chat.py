"""Ollama(gemma4 등 tool-calling 모델)로 자연어 질문 → 예측 도구 호출 → 답변 (Phase 7).

실행
  uv run python src/ollama_chat.py                                   # 대화형 (종료: 빈 줄 또는 q)
  uv run python src/ollama_chat.py --once "추정가격 10억 공사, 예가 범위 ±3%, 타 기관 입찰이야. 투찰가 알려줘"
  uv run python src/ollama_chat.py --model gemma4:latest --once "..."    # 8B: 빠르지만(약 30초) 인자 누락·되묻기가 잦음

LLM 은 질문을 해석해 bid_tools 의 함수를 호출하고 결과를 설명하는 인터페이스다. 숫자를 스스로 만들지 않도록
시스템 프롬프트로 묶고, 도구 결과를 그대로 대화에 넣는다. Ollama 서버(localhost:11434)가 떠 있어야 한다.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bid_tools as B  # noqa: E402

DEFAULT_MODEL = "gemma4:26b"      # 8B(gemma4:latest)는 인자를 빠뜨리거나 되묻는 일이 잦아 26B 를 기본으로 한다 (실측 2026-10-18)
MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = """너는 공공입찰 예정가격(사정률) 예측 도구의 안내자다. 한국어로 답한다.
규칙:
1. 예정가격·사정률·투찰가·확률 등 모든 숫자는 반드시 도구(predict_bid, evaluate_bid, compare_manual, backtest_summary, model_info)의 결과에서만 가져온다. 스스로 계산하거나 추정하지 않는다.
2. 타 기관(남동발전이 아닌 공공기관) 입찰이면 market=true 이고 range_pct(공고문의 '기초금액 ±N% 범위' N, 2 또는 3)가 필요하다. 이것만 사용자가 말하지 않았으면 되묻되, "대부분 3"이라고 알려준다. 남동발전 사내 발주면 market=false.
   낙찰자결정방법·기준 분위수·낙찰하한율·안전마진·참여업체수·부서는 **절대 되묻지 않는다**. 추정가격(과 타 기관이면 예가 범위)만 있으면 즉시 predict_bid_simple 을 호출한다. 사용자가 그 조건을 직접 말한 경우에만 predict_bid 에 넣는다.
   기본 권장 기준은 Q75(통과확률 약 75%)다. Q50 을 기본으로 권하지 않는다.
3. 금액은 원 단위 천 단위 구분으로, 사정률은 소수 3자리 %로 쓴다. 답에는 기초금액, Q25/Q50/Q75 사정률과 예정가격, 권장 투찰가(기준 분위수와 통과확률), 그리고 "Qα 기준 투찰가는 실제 사정률이 Qα 이하일 때만 유효(통과확률≈α)"라는 주의를 넣는다.
4. 도구가 error 를 돌려주면 그 내용을 그대로 전달하고 필요한 입력을 묻는다.
5. 짧고 명확하게 답한다."""


def _coerce(fn, args: dict) -> dict:
    """LLM 이 문자열로 준 인자를 함수 시그니처의 타입으로 맞춘다 (int/float/bool)."""
    out = {}
    params = inspect.signature(fn).parameters
    for k, v in (args or {}).items():
        if k not in params:
            continue
        ann = str(params[k].annotation)
        if v is None or v == "":
            out[k] = None
            continue
        try:
            if "bool" in ann:
                out[k] = v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "y", "예", "참")
            elif "int" in ann and "float" not in ann:
                out[k] = int(float(v))
            elif "float" in ann:
                out[k] = float(str(v).replace(",", "").replace("원", ""))
            else:
                out[k] = v
        except (TypeError, ValueError):
            out[k] = v
    return out


def run_turn(client, model: str, messages: list, tools=None, dispatch: dict | None = None,
             max_rounds: int = MAX_TOOL_ROUNDS, think: bool | None = False, log=None) -> str:
    """메시지 목록에 대해 도구 호출 루프를 돌리고 최종 답변 문자열을 돌려준다. messages 는 제자리에서 갱신된다."""
    tools = B.TOOLS if tools is None else tools
    dispatch = B.DISPATCH if dispatch is None else dispatch
    for _ in range(max_rounds):
        resp = client.chat(model=model, messages=messages, tools=tools, think=think)
        msg = resp.message
        calls = list(msg.tool_calls or [])
        if not calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            return msg.content or ""
        messages.append(msg)
        for call in calls:
            name = call.function.name
            fn = dispatch.get(name)
            if fn is None:
                result = {"error": f"알 수 없는 도구 {name}"}
            else:
                kwargs = _coerce(fn, dict(call.function.arguments or {}))
                if log:
                    log(f"[도구 호출] {name}({json.dumps(kwargs, ensure_ascii=False)})")
                try:
                    result = fn(**kwargs)
                except Exception as e:  # noqa: BLE001 — LLM 에 오류를 돌려줘 되묻게 한다
                    result = {"error": f"{type(e).__name__}: {e}"}
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result, ensure_ascii=False, default=str)})
    # 도구 호출만 반복하면 마지막으로 정리 답변을 받는다
    resp = client.chat(model=model, messages=messages, think=think)
    messages.append({"role": "assistant", "content": resp.message.content or ""})
    return resp.message.content or ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ollama 자연어 인터페이스")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"tool calling 을 지원하는 Ollama 모델 (기본 {DEFAULT_MODEL})")
    ap.add_argument("--once", help="한 질문만 처리하고 종료")
    ap.add_argument("--host", default=None, help="Ollama 서버 (기본 http://localhost:11434)")
    ap.add_argument("--think", action="store_true", help="모델 thinking 켜기 (느림)")
    ap.add_argument("--verbose", action="store_true", help="도구 호출 내용 출력")
    args = ap.parse_args(argv)

    import ollama
    client = ollama.Client(host=args.host) if args.host else ollama.Client()
    try:
        caps = client.show(args.model).capabilities or []
    except Exception as e:  # noqa: BLE001
        print(f"Ollama 에 연결하거나 모델을 찾을 수 없습니다: {e}", file=sys.stderr)
        return 1
    if "tools" not in caps:
        print(f"모델 {args.model} 은 tool calling 을 지원하지 않습니다 (capabilities={caps}). gemma4 등을 쓰세요.", file=sys.stderr)
        return 1

    log = print if args.verbose else None
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if args.once:
        messages.append({"role": "user", "content": args.once})
        print(run_turn(client, args.model, messages, think=args.think or None, log=log))
        return 0
    print(f"모델 {args.model}. 질문을 입력하세요 (빈 줄 또는 q 로 종료).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("q", "quit", "exit"):
            break
        messages.append({"role": "user", "content": q})
        print(run_turn(client, args.model, messages, think=args.think or None, log=log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
