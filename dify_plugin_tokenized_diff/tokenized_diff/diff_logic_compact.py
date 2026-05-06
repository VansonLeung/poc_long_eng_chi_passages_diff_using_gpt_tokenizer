from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

from diff_logic import (
    ENCODING_NAME,
    build_token_replacements,
    build_string_replacement_mapping,
    LineReplacement,
    TokenReplacement,
    decode_tokens,
    tokenize,
)
from diff_logic_gitlike import build_gitlike_line_replacement_mapping


LONG_LINE_THRESHOLD_CHARS = 100
CONTEXT_WINDOW_CHARS = 30
PREVIEW_CHARS = 40
MERGE_SPAN_GAP_CHARS = 24
WINDOW_JOIN_GAP_CHARS = 0
CONSECUTIVE_NEWLINE_PATTERN = re.compile(r"\n{2,}")
DASH_FAMILY_CLASS = r"\-\u2010\u2011"
DASH_SPACING_PATTERN = re.compile(rf"([{DASH_FAMILY_CLASS}])(\s+)(?=\w)")


@dataclass
class CompactSpan:
    index: int
    opcodes: list[str]
    original_char_start: int
    original_char_end: int
    masked_char_start: int
    masked_char_end: int
    original_text: str
    masked_text: str
    original_context: str
    masked_context: str
    original_fragments: list[str] = field(default_factory=list)
    masked_fragments: list[str] = field(default_factory=list)


def flatten_text_to_single_line(text: str) -> str:
    return " ".join(text.split())


def normalize_newlines_by_script(text: str) -> str:
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_text = CONSECUTIVE_NEWLINE_PATTERN.sub("\n", normalized_text)

    output_parts: list[str] = []
    for index, char in enumerate(normalized_text):
        if char != "\n":
            output_parts.append(char)
            continue

        left_kind = find_adjacent_script_kind(normalized_text, index, step=-1)
        right_kind = find_adjacent_script_kind(normalized_text, index, step=1)
        output_parts.append(script_joiner_for_newline(left_kind, right_kind))

    return "".join(output_parts)


def build_script_normalized_compact_analysis(
    original_text: str,
    masked_text: str,
    *,
    encoding_name: str = ENCODING_NAME,
    context_window_chars: int = CONTEXT_WINDOW_CHARS,
    preview_chars: int = PREVIEW_CHARS,
    window_join_gap_chars: int = WINDOW_JOIN_GAP_CHARS,
) -> dict:
    normalized_original_text = normalize_newlines_by_script(original_text)
    normalized_masked_text = normalize_newlines_by_script(masked_text)
    return build_normalized_compact_analysis(
        original_text=original_text,
        masked_text=masked_text,
        normalized_original_text=normalized_original_text,
        normalized_masked_text=normalized_masked_text,
        encoding_name=encoding_name,
        context_window_chars=context_window_chars,
        preview_chars=preview_chars,
        window_join_gap_chars=window_join_gap_chars,
        analysis_mode="script_normalized_compact_for_llm",
        normalization_strategy="collapse_consecutive_newlines_to_one_then_convert_single_newlines_by_adjacent_english_chinese_chars",
    )


def build_flattened_compact_analysis(
    original_text: str,
    masked_text: str,
    *,
    encoding_name: str = ENCODING_NAME,
    context_window_chars: int = CONTEXT_WINDOW_CHARS,
    preview_chars: int = PREVIEW_CHARS,
    window_join_gap_chars: int = WINDOW_JOIN_GAP_CHARS,
) -> dict:
    normalized_original_text = flatten_text_to_single_line(original_text)
    normalized_masked_text = flatten_text_to_single_line(masked_text)
    return build_normalized_compact_analysis(
        original_text=original_text,
        masked_text=masked_text,
        normalized_original_text=normalized_original_text,
        normalized_masked_text=normalized_masked_text,
        encoding_name=encoding_name,
        context_window_chars=context_window_chars,
        preview_chars=preview_chars,
        window_join_gap_chars=window_join_gap_chars,
        analysis_mode="flattened_compact_for_llm",
        normalization_strategy="collapse_all_whitespace_to_single_spaces",
    )


def build_normalized_compact_analysis(
    *,
    original_text: str,
    masked_text: str,
    normalized_original_text: str,
    normalized_masked_text: str,
    encoding_name: str,
    context_window_chars: int,
    preview_chars: int,
    window_join_gap_chars: int,
    analysis_mode: str,
    normalization_strategy: str,
) -> dict:
    raw_result = build_string_replacement_mapping(
        normalized_original_text,
        normalized_masked_text,
        encoding_name=encoding_name,
    )
    filtered_replacements = [
        item
        for item in raw_result["replacements"]
        if not is_ignorable_noise_replacement(item)
    ]

    compact_windows = build_compact_spans_from_replacements(
        normalized_original_text,
        normalized_masked_text,
        filtered_replacements,
        encoding_name=encoding_name,
        context_window_chars=context_window_chars,
        window_join_gap_chars=window_join_gap_chars,
    )

    return {
        "encoding": encoding_name,
        "analysis_mode": analysis_mode,
        "payload_mode": "context_windows_only",
        "alignment_strategy": "whole_text_token_diff_then_context_window_merge",
        "normalization_strategy": normalization_strategy,
        "original_input_char_count": len(original_text),
        "masked_input_char_count": len(masked_text),
        "original_normalized_char_count": len(normalized_original_text),
        "masked_normalized_char_count": len(normalized_masked_text),
        "original_token_count": raw_result["original_token_count"],
        "masked_token_count": raw_result["masked_token_count"],
        "replacement_count": len(filtered_replacements),
        "ignored_replacement_count": raw_result["replacement_count"] - len(filtered_replacements),
        "change_window_count": len(compact_windows),
        "overall_similarity": line_similarity(normalized_original_text, normalized_masked_text),
        "context_window_chars": context_window_chars,
        "window_join_gap_chars": window_join_gap_chars,
        "original_preview_head": preview_head(normalized_original_text, preview_chars),
        "original_preview_tail": preview_tail(normalized_original_text, preview_chars),
        "masked_preview_head": preview_head(normalized_masked_text, preview_chars),
        "masked_preview_tail": preview_tail(normalized_masked_text, preview_chars),
        "change_windows": [asdict(span) for span in compact_windows],
    }


def is_ignorable_noise_replacement(replacement: dict) -> bool:
    return is_whitespace_only_replacement(replacement) or is_ignorable_dash_spacing_replacement(replacement)


def is_whitespace_only_replacement(replacement: dict) -> bool:
    original_text = replacement["original_text"]
    masked_text = replacement["masked_text"]
    return normalize_all_whitespace(original_text) == normalize_all_whitespace(masked_text)


def is_ignorable_dash_spacing_replacement(replacement: dict) -> bool:
    original_text = replacement["original_text"]
    masked_text = replacement["masked_text"]
    if original_text == masked_text:
        return False
    normalized_original = normalize_dash_spacing(original_text)
    normalized_masked = normalize_dash_spacing(masked_text)
    return normalized_original == normalized_masked


def normalize_dash_spacing(text: str) -> str:
    return DASH_SPACING_PATTERN.sub(r"\1", text)


def normalize_all_whitespace(text: str) -> str:
    return " ".join(text.split())


def find_adjacent_script_kind(text: str, newline_index: int, *, step: int) -> str | None:
    cursor = newline_index + step
    while 0 <= cursor < len(text):
        candidate = text[cursor]
        if candidate == "\n" or candidate.isspace() or is_ignorable_boundary_punctuation(candidate):
            cursor += step
            continue
        if is_english_char(candidate):
            return "english"
        if is_chinese_char(candidate):
            return "chinese"
        cursor += step
    return None


def script_joiner_for_newline(left_kind: str | None, right_kind: str | None) -> str:
    if left_kind == "chinese" and right_kind == "chinese":
        return ""
    return " "


def is_ignorable_boundary_punctuation(char: str) -> bool:
    return unicodedata.category(char).startswith("P")


def is_english_char(char: str) -> bool:
    return char.isascii() and char.isalpha()


def is_chinese_char(char: str) -> bool:
    code_point = ord(char)
    return (
        0x3400 <= code_point <= 0x4DBF
        or 0x4E00 <= code_point <= 0x9FFF
        or 0xF900 <= code_point <= 0xFAFF
        or 0x20000 <= code_point <= 0x2A6DF
        or 0x2A700 <= code_point <= 0x2B73F
        or 0x2B740 <= code_point <= 0x2B81F
        or 0x2B820 <= code_point <= 0x2CEAF
        or 0x2CEB0 <= code_point <= 0x2EBEF
        or 0x30000 <= code_point <= 0x3134F
    )


def build_gitlike_compact_analysis(
    original_text: str,
    masked_text: str,
    *,
    encoding_name: str = ENCODING_NAME,
    long_line_threshold_chars: int = LONG_LINE_THRESHOLD_CHARS,
    context_window_chars: int = CONTEXT_WINDOW_CHARS,
    preview_chars: int = PREVIEW_CHARS,
) -> dict:
    raw_result = build_gitlike_line_replacement_mapping(
        original_text,
        masked_text,
        encoding_name=encoding_name,
    )
    return compact_gitlike_result(
        raw_result,
        encoding_name=encoding_name,
        long_line_threshold_chars=long_line_threshold_chars,
        context_window_chars=context_window_chars,
        preview_chars=preview_chars,
    )


def compact_gitlike_result(
    raw_result: dict,
    *,
    encoding_name: str = ENCODING_NAME,
    long_line_threshold_chars: int = LONG_LINE_THRESHOLD_CHARS,
    context_window_chars: int = CONTEXT_WINDOW_CHARS,
    preview_chars: int = PREVIEW_CHARS,
) -> dict:
    compact_hunks = []
    total_long_line_operations = 0
    total_compact_spans = 0

    for hunk in raw_result["hunks"]:
        compact_operations = []
        for operation in hunk["operations"]:
            compact_operation = compact_operation_for_llm(
                operation,
                encoding_name=encoding_name,
                long_line_threshold_chars=long_line_threshold_chars,
                context_window_chars=context_window_chars,
                preview_chars=preview_chars,
            )
            if compact_operation["payload_mode"] == "context_windows_only":
                total_long_line_operations += 1
            total_compact_spans += compact_operation["change_span_count"]
            compact_operations.append(compact_operation)

        compact_hunks.append(
            {
                "index": hunk["index"],
                "original_line_start": hunk["original_line_start"],
                "original_line_end": hunk["original_line_end"],
                "masked_line_start": hunk["masked_line_start"],
                "masked_line_end": hunk["masked_line_end"],
                "operation_count": len(compact_operations),
                "operations": compact_operations,
            }
        )

    return {
        "encoding": raw_result["encoding"],
        "line_filter": raw_result["line_filter"],
        "alignment_strategy": raw_result["alignment_strategy"],
        "analysis_mode": "compact_for_llm",
        "original_line_count": raw_result["original_line_count"],
        "masked_line_count": raw_result["masked_line_count"],
        "replacement_count": raw_result["replacement_count"],
        "hunk_count": raw_result["hunk_count"],
        "long_line_threshold_chars": long_line_threshold_chars,
        "context_window_chars": context_window_chars,
        "total_long_line_operations": total_long_line_operations,
        "total_compact_spans": total_compact_spans,
        "hunks": compact_hunks,
    }


def compact_operation_for_llm(
    operation: dict,
    *,
    encoding_name: str,
    long_line_threshold_chars: int,
    context_window_chars: int,
    preview_chars: int,
) -> dict:
    original_line = operation["original_lines"][0] if operation["original_lines"] else ""
    masked_line = operation["masked_lines"][0] if operation["masked_lines"] else ""
    original_length = len(original_line)
    masked_length = len(masked_line)
    is_long_line = max(original_length, masked_length) >= long_line_threshold_chars
    similarity = line_similarity(original_line, masked_line)
    compact_spans = build_compact_spans(
        original_line,
        masked_line,
        [TokenReplacement(**item) for item in operation.get("token_replacements", [])],
        encoding_name=encoding_name,
        context_window_chars=context_window_chars,
    )

    payload_mode = "context_windows_only" if is_long_line else "full_lines"
    compact_operation = {
        "index": operation["index"],
        "opcode": operation["opcode"],
        "original_line_start": operation["original_line_start"],
        "original_line_end": operation["original_line_end"],
        "masked_line_start": operation["masked_line_start"],
        "masked_line_end": operation["masked_line_end"],
        "payload_mode": payload_mode,
        "line_similarity": similarity,
        "original_line_length": original_length,
        "masked_line_length": masked_length,
        "change_span_count": len(compact_spans),
        "change_spans": [asdict(span) for span in compact_spans],
    }

    if payload_mode == "full_lines":
        compact_operation["original_lines"] = operation["original_lines"]
        compact_operation["masked_lines"] = operation["masked_lines"]
    else:
        compact_operation["original_preview_head"] = preview_head(original_line, preview_chars)
        compact_operation["original_preview_tail"] = preview_tail(original_line, preview_chars)
        compact_operation["masked_preview_head"] = preview_head(masked_line, preview_chars)
        compact_operation["masked_preview_tail"] = preview_tail(masked_line, preview_chars)

    return compact_operation


def build_compact_spans(
    original_line: str,
    masked_line: str,
    token_replacements: list[TokenReplacement],
    *,
    encoding_name: str,
    context_window_chars: int,
) -> list[CompactSpan]:
    if not token_replacements:
        if original_line == masked_line:
            return []
        return [
            CompactSpan(
                index=0,
                opcodes=["replace"],
                original_char_start=0,
                original_char_end=len(original_line),
                masked_char_start=0,
                masked_char_end=len(masked_line),
                original_text=original_line,
                masked_text=masked_line,
                original_context=preview_head(original_line, context_window_chars * 2),
                masked_context=preview_head(masked_line, context_window_chars * 2),
            )
        ]

    raw_spans = [
        token_replacement_to_span(
            original_line,
            masked_line,
            token_replacement,
            encoding_name=encoding_name,
            context_window_chars=context_window_chars,
        )
        for token_replacement in token_replacements
    ]
    return merge_compact_spans(
        raw_spans,
        original_line,
        masked_line,
        encoding_name,
        context_window_chars,
    )


def build_compact_spans_from_replacements(
    original_text: str,
    masked_text: str,
    replacements: list[dict],
    *,
    encoding_name: str,
    context_window_chars: int,
    window_join_gap_chars: int,
) -> list[CompactSpan]:
    if not replacements:
        return []

    raw_spans = [
        replacement_to_span(
            original_text,
            masked_text,
            replacement,
            encoding_name=encoding_name,
            context_window_chars=context_window_chars,
        )
        for replacement in replacements
    ]
    return merge_compact_spans_by_window(
        raw_spans,
        original_text,
        masked_text,
        encoding_name=encoding_name,
        context_window_chars=context_window_chars,
        window_join_gap_chars=window_join_gap_chars,
    )


def replacement_to_span(
    original_text: str,
    masked_text: str,
    replacement: dict,
    *,
    encoding_name: str,
    context_window_chars: int,
) -> CompactSpan:
    original_start, original_end = token_offsets_to_char_range(
        original_text,
        replacement["original_token_start"],
        replacement["original_token_end"],
        encoding_name,
    )
    masked_start, masked_end = token_offsets_to_char_range(
        masked_text,
        replacement["masked_token_start"],
        replacement["masked_token_end"],
        encoding_name,
    )

    return CompactSpan(
        index=replacement["index"],
        opcodes=[replacement["opcode"]],
        original_char_start=original_start,
        original_char_end=original_end,
        masked_char_start=masked_start,
        masked_char_end=masked_end,
        original_text=original_text[original_start:original_end],
        masked_text=masked_text[masked_start:masked_end],
        original_context=extract_context(
            original_text,
            original_start,
            original_end,
            context_window_chars,
        ),
        masked_context=extract_context(
            masked_text,
            masked_start,
            masked_end,
            context_window_chars,
        ),
        original_fragments=[original_text[original_start:original_end]] if original_text[original_start:original_end] else [],
        masked_fragments=[masked_text[masked_start:masked_end]] if masked_text[masked_start:masked_end] else [],
    )


def token_replacement_to_span(
    original_line: str,
    masked_line: str,
    token_replacement: TokenReplacement,
    *,
    encoding_name: str,
    context_window_chars: int,
) -> CompactSpan:
    original_start, original_end = token_offsets_to_char_range(
        original_line,
        token_replacement.original_token_start,
        token_replacement.original_token_end,
        encoding_name,
    )
    masked_start, masked_end = token_offsets_to_char_range(
        masked_line,
        token_replacement.masked_token_start,
        token_replacement.masked_token_end,
        encoding_name,
    )

    return CompactSpan(
        index=0,
        opcodes=[token_replacement.opcode],
        original_char_start=original_start,
        original_char_end=original_end,
        masked_char_start=masked_start,
        masked_char_end=masked_end,
        original_text=original_line[original_start:original_end],
        masked_text=masked_line[masked_start:masked_end],
        original_context=extract_context(original_line, original_start, original_end, context_window_chars),
        masked_context=extract_context(masked_line, masked_start, masked_end, context_window_chars),
        original_fragments=[original_line[original_start:original_end]] if original_line[original_start:original_end] else [],
        masked_fragments=[masked_line[masked_start:masked_end]] if masked_line[masked_start:masked_end] else [],
    )


def merge_compact_spans(
    spans: list[CompactSpan],
    original_line: str,
    masked_line: str,
    encoding_name: str,
    context_window_chars: int,
) -> list[CompactSpan]:
    if not spans:
        return []

    merged: list[CompactSpan] = []
    current = spans[0]

    for next_span in spans[1:]:
        original_gap = next_span.original_char_start - current.original_char_end
        masked_gap = next_span.masked_char_start - current.masked_char_end
        if original_gap <= MERGE_SPAN_GAP_CHARS and masked_gap <= MERGE_SPAN_GAP_CHARS:
            current = CompactSpan(
                index=0,
                opcodes=dedupe_preserve_order(current.opcodes + next_span.opcodes),
                original_char_start=min(current.original_char_start, next_span.original_char_start),
                original_char_end=max(current.original_char_end, next_span.original_char_end),
                masked_char_start=min(current.masked_char_start, next_span.masked_char_start),
                masked_char_end=max(current.masked_char_end, next_span.masked_char_end),
                original_text="",
                masked_text="",
                original_context="",
                masked_context="",
                original_fragments=current.original_fragments + next_span.original_fragments,
                masked_fragments=current.masked_fragments + next_span.masked_fragments,
            )
            continue

        merged.append(
            finalize_compact_span(
                current,
                original_line,
                masked_line,
                encoding_name,
                context_window_chars,
                len(merged),
            )
        )
        current = next_span

    merged.append(
        finalize_compact_span(
            current,
            original_line,
            masked_line,
            encoding_name,
            context_window_chars,
            len(merged),
        )
    )
    return merged


def merge_compact_spans_by_window(
    spans: list[CompactSpan],
    original_text: str,
    masked_text: str,
    *,
    encoding_name: str,
    context_window_chars: int,
    window_join_gap_chars: int,
) -> list[CompactSpan]:
    if not spans:
        return []

    merged: list[CompactSpan] = []
    current = spans[0]
    current_original_window = expanded_window_bounds(
        current.original_char_start,
        current.original_char_end,
        len(original_text),
        context_window_chars,
    )
    current_masked_window = expanded_window_bounds(
        current.masked_char_start,
        current.masked_char_end,
        len(masked_text),
        context_window_chars,
    )

    for next_span in spans[1:]:
        next_original_window = expanded_window_bounds(
            next_span.original_char_start,
            next_span.original_char_end,
            len(original_text),
            context_window_chars,
        )
        next_masked_window = expanded_window_bounds(
            next_span.masked_char_start,
            next_span.masked_char_end,
            len(masked_text),
            context_window_chars,
        )

        windows_touch = (
            next_original_window[0] <= current_original_window[1] + window_join_gap_chars
            and next_masked_window[0] <= current_masked_window[1] + window_join_gap_chars
        )
        if windows_touch:
            current = CompactSpan(
                index=current.index,
                opcodes=dedupe_preserve_order(current.opcodes + next_span.opcodes),
                original_char_start=min(current.original_char_start, next_span.original_char_start),
                original_char_end=max(current.original_char_end, next_span.original_char_end),
                masked_char_start=min(current.masked_char_start, next_span.masked_char_start),
                masked_char_end=max(current.masked_char_end, next_span.masked_char_end),
                original_text="",
                masked_text="",
                original_context="",
                masked_context="",
                original_fragments=current.original_fragments + next_span.original_fragments,
                masked_fragments=current.masked_fragments + next_span.masked_fragments,
            )
            current_original_window = (
                min(current_original_window[0], next_original_window[0]),
                max(current_original_window[1], next_original_window[1]),
            )
            current_masked_window = (
                min(current_masked_window[0], next_masked_window[0]),
                max(current_masked_window[1], next_masked_window[1]),
            )
            continue

        merged.append(
            finalize_compact_span(
                current,
                original_text,
                masked_text,
                encoding_name,
                context_window_chars,
                len(merged),
            )
        )
        current = next_span
        current_original_window = next_original_window
        current_masked_window = next_masked_window

    merged.append(
        finalize_compact_span(
            current,
            original_text,
            masked_text,
            encoding_name,
            context_window_chars,
            len(merged),
        )
    )
    return merged


def expanded_window_bounds(
    start_char: int,
    end_char: int,
    text_length: int,
    context_window_chars: int,
) -> tuple[int, int]:
    return (
        max(0, start_char - context_window_chars),
        min(text_length, end_char + context_window_chars),
    )


def finalize_compact_span(
    span: CompactSpan,
    original_line: str,
    masked_line: str,
    encoding_name: str,
    context_window_chars: int,
    index: int,
) -> CompactSpan:
    original_context_left, original_context_right = expanded_window_bounds(
        span.original_char_start,
        span.original_char_end,
        len(original_line),
        context_window_chars,
    )
    masked_context_left, masked_context_right = expanded_window_bounds(
        span.masked_char_start,
        span.masked_char_end,
        len(masked_line),
        context_window_chars,
    )
    window_original_text = original_line[original_context_left:original_context_right]
    window_masked_text = masked_line[masked_context_left:masked_context_right]
    original_fragments, masked_fragments = extract_changed_fragments(
        window_original_text,
        window_masked_text,
        encoding_name,
    )

    return CompactSpan(
        index=index,
        opcodes=span.opcodes,
        original_char_start=span.original_char_start,
        original_char_end=span.original_char_end,
        masked_char_start=span.masked_char_start,
        masked_char_end=span.masked_char_end,
        original_text=join_changed_fragments(original_fragments),
        masked_text=join_changed_fragments(masked_fragments),
        original_context=extract_context(
            original_line,
            span.original_char_start,
            span.original_char_end,
            context_window_chars,
        ),
        masked_context=extract_context(
            masked_line,
            span.masked_char_start,
            span.masked_char_end,
            context_window_chars,
        ),
        original_fragments=original_fragments,
        masked_fragments=masked_fragments,
    )


def extract_changed_fragments(
    original_text: str,
    masked_text: str,
    encoding_name: str,
) -> tuple[list[str], list[str]]:
    replacements = build_token_replacements(original_text, masked_text, encoding_name)
    if not replacements:
        original_fragment = original_text.strip()
        masked_fragment = masked_text.strip()
        return (
            [original_fragment] if original_fragment and original_fragment != masked_fragment else [],
            [masked_fragment] if masked_fragment and masked_fragment != original_fragment else [],
        )

    original_fragments = normalize_changed_fragments(item.original_text for item in replacements)
    masked_fragments = normalize_changed_fragments(item.masked_text for item in replacements)
    return original_fragments, masked_fragments


def normalize_changed_fragments(fragments: list[str] | tuple[str, ...] | object) -> list[str]:
    normalized: list[str] = []
    for fragment in fragments:
        stripped_fragment = fragment.strip()
        if not stripped_fragment:
            continue
        normalized.append(stripped_fragment)
    return normalized


def join_changed_fragments(fragments: list[str]) -> str:
    return " | ".join(fragments)


def token_offsets_to_char_range(
    line: str,
    start_token: int,
    end_token: int,
    encoding_name: str,
) -> tuple[int, int]:
    tokens = tokenize(line, encoding_name)
    start_char = len(decode_tokens(tokens[:start_token], encoding_name))
    end_char = len(decode_tokens(tokens[:end_token], encoding_name))
    return start_char, end_char


def extract_context(line: str, start_char: int, end_char: int, context_window_chars: int) -> str:
    left = max(0, start_char - context_window_chars)
    right = min(len(line), end_char + context_window_chars)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(line) else ""
    return f"{prefix}{line[left:right]}{suffix}"


def preview_head(line: str, preview_chars: int) -> str:
    if len(line) <= preview_chars:
        return line
    return f"{line[:preview_chars]}..."


def preview_tail(line: str, preview_chars: int) -> str:
    if len(line) <= preview_chars:
        return line
    return f"...{line[-preview_chars:]}"


def line_similarity(original_line: str, masked_line: str) -> float:
    if not original_line and not masked_line:
        return 1.0
    return SequenceMatcher(a=original_line, b=masked_line, autojunk=False).ratio()


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped