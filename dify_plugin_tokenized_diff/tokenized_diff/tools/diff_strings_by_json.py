from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from diff_logic import build_json_replacement_mapping, stringify_result


class DiffStringsByJsonTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        stringified_json_a = str(tool_parameters.get("stringified_json_a") or "")
        stringified_json_b = str(tool_parameters.get("stringified_json_b") or "")
        result = build_json_replacement_mapping(stringified_json_a, stringified_json_b)
        yield self.create_text_message(stringify_result(result))