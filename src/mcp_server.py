"""MCP 서버 (Phase 7): 예정가격 예측 도구를 Claude Code·Claude Desktop 등 MCP 클라이언트에 노출한다.

실행: uv run python src/mcp_server.py           (stdio)
프로젝트 .mcp.json 에 등록되어 있어 Claude Code 에서는 자동으로 뜬다.
도구 구현은 bid_tools.py 에 있고, 숫자는 전부 predict_cli/evaluate 가 계산한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bid_tools as B  # noqa: E402

from mcp.server.mcpserver import MCPServer  # noqa: E402

server = MCPServer(
    name="yega-predictor",
    instructions=(
        "공공입찰 예정가격(사정률) 예측 도구. predict_bid 로 추정가격→Q25/Q50/Q75 예정가격·투찰가, "
        "evaluate_bid/compare_manual 로 실제값 대비 검증, backtest_summary 로 홀드아웃 성적, model_info 로 모델 정보. "
        "타 기관 입찰은 market=True 와 range_pct(공고의 예가 범위 2 또는 3)를 넣는다. 숫자는 도구 결과를 그대로 전달한다."
    ),
)

for fn in B.TOOLS:
    server.tool()(fn)

if __name__ == "__main__":
    server.run("stdio")
