from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from diff_logic import stringify_result
from diff_logic_compact import build_script_normalized_compact_analysis


class DiffStringsScriptNormalizedCompactTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        string_a = str(tool_parameters.get("string_a") or "")
        string_b = str(tool_parameters.get("string_b") or "")
        context_window_chars = int(tool_parameters.get("context_window_chars") or 30)
        window_join_gap_chars = int(tool_parameters.get("window_join_gap_chars") or 0)
        result = build_script_normalized_compact_analysis(
            string_a,
            string_b,
            context_window_chars=context_window_chars,
            window_join_gap_chars=window_join_gap_chars,
        )
        yield self.create_text_message(stringify_result(result))