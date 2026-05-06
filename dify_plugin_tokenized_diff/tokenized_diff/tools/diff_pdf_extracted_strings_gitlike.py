from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from diff_logic import stringify_result
from diff_logic_pdf import build_pdf_extract_gitlike_replacement_mapping


class DiffPdfExtractedStringsGitlikeTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        string_a = str(tool_parameters.get("string_a") or "")
        string_b = str(tool_parameters.get("string_b") or "")
        result = build_pdf_extract_gitlike_replacement_mapping(string_a, string_b)
        yield self.create_text_message(stringify_result(result))