from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

import tiktoken


ENCODING_NAME = "cl100k_base"
CHUNK_SIZE_TOKENS = 120
TRAILING_PUNCTUATION = " \t\r\n,.;:!?。．，；：！？、\"'"


@dataclass
class Replacement:
    index: int
    opcode: str
    original_token_start: int
    original_token_end: int
    masked_token_start: int
    masked_token_end: int
    original_chunk: int
    masked_chunk: int
    original_text: str
    masked_text: str
    original_tokens: list[int]
    masked_tokens: list[int]


@dataclass
class TokenReplacement:
    index: int
    opcode: str
    original_token_start: int
    original_token_end: int
    masked_token_start: int
    masked_token_end: int
    original_text: str
    masked_text: str
    original_tokens: list[int]
    masked_tokens: list[int]


@dataclass
class LineReplacement:
    index: int
    opcode: str
    original_line_start: int
    original_line_end: int
    masked_line_start: int
    masked_line_end: int
    original_lines: list[str]
    masked_lines: list[str]
    token_replacements: list[TokenReplacement]


def get_encoding(encoding_name: str = ENCODING_NAME):
    return tiktoken.get_encoding(encoding_name)


def tokenize(text: str, encoding_name: str = ENCODING_NAME) -> list[int]:
    encoding = get_encoding(encoding_name)
    return encoding.encode(text)


def decode_tokens(tokens: list[int], encoding_name: str = ENCODING_NAME) -> str:
    if not tokens:
        return ""
    encoding = get_encoding(encoding_name)
    return encoding.decode(tokens)


def should_merge_equal_gap(text: str) -> bool:
    if not text:
        return False
    if any(char.isspace() for char in text):
        return False
    return True


def split_non_empty_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def normalize_replacement_value(value: str) -> str:
    return value.strip().rstrip(TRAILING_PUNCTUATION).strip()


def parse_json_lines(raw_text: str) -> list[str]:
    raw = json.loads(raw_text)
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
        return [str(item) for item in raw[0]]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    raise ValueError("Unsupported JSON structure. Expected a JSON array of strings.")


def build_string_replacement_mapping(
    original_text: str,
    masked_text: str,
    *,
    encoding_name: str = ENCODING_NAME,
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
) -> dict:
    original_tokens = tokenize(original_text, encoding_name)
    masked_tokens = tokenize(masked_text, encoding_name)

    matcher = SequenceMatcher(a=original_tokens, b=masked_tokens, autojunk=False)
    replacements: list[Replacement] = []

    for index, (opcode, a0, a1, b0, b1) in enumerate(matcher.get_opcodes()):
        if opcode == "equal":
            continue

        replacements.append(
            Replacement(
                index=index,
                opcode=opcode,
                original_token_start=a0,
                original_token_end=a1,
                masked_token_start=b0,
                masked_token_end=b1,
                original_chunk=a0 // chunk_size_tokens,
                masked_chunk=b0 // chunk_size_tokens,
                original_text=decode_tokens(original_tokens[a0:a1], encoding_name),
                masked_text=decode_tokens(masked_tokens[b0:b1], encoding_name),
                original_tokens=original_tokens[a0:a1],
                masked_tokens=masked_tokens[b0:b1],
            )
        )

    return {
        "encoding": encoding_name,
        "chunk_size_tokens": chunk_size_tokens,
        "original_token_count": len(original_tokens),
        "masked_token_count": len(masked_tokens),
        "replacement_count": len(replacements),
        "replacements": [asdict(item) for item in replacements],
    }


def build_token_replacements(
    original_line: str,
    masked_line: str,
    encoding_name: str = ENCODING_NAME,
) -> list[TokenReplacement]:
    original_tokens = tokenize(original_line, encoding_name)
    masked_tokens = tokenize(masked_line, encoding_name)
    matcher = SequenceMatcher(a=original_tokens, b=masked_tokens, autojunk=False)
    opcodes = matcher.get_opcodes()
    replacements: list[TokenReplacement] = []
    opcode_index = 0
    replacement_index = 0

    while opcode_index < len(opcodes):
        opcode, a0, a1, b0, b1 = opcodes[opcode_index]
        if opcode == "equal":
            opcode_index += 1
            continue

        merged_a0 = a0
        merged_a1 = a1
        merged_b0 = b0
        merged_b1 = b1
        merged_opcode = opcode

        while opcode_index + 2 < len(opcodes):
            gap_opcode, gap_a0, gap_a1, gap_b0, gap_b1 = opcodes[opcode_index + 1]
            next_opcode, _, next_a1, _, next_b1 = opcodes[opcode_index + 2]
            if gap_opcode != "equal" or next_opcode == "equal":
                break

            gap_original_text = decode_tokens(original_tokens[gap_a0:gap_a1], encoding_name)
            gap_masked_text = decode_tokens(masked_tokens[gap_b0:gap_b1], encoding_name)
            if gap_original_text != gap_masked_text:
                break
            if not should_merge_equal_gap(gap_original_text):
                break

            merged_a1 = next_a1
            merged_b1 = next_b1
            merged_opcode = "replace"
            opcode_index += 2

        replacements.append(
            TokenReplacement(
                index=replacement_index,
                opcode=merged_opcode,
                original_token_start=merged_a0,
                original_token_end=merged_a1,
                masked_token_start=merged_b0,
                masked_token_end=merged_b1,
                original_text=decode_tokens(original_tokens[merged_a0:merged_a1], encoding_name),
                masked_text=decode_tokens(masked_tokens[merged_b0:merged_b1], encoding_name),
                original_tokens=original_tokens[merged_a0:merged_a1],
                masked_tokens=masked_tokens[merged_b0:merged_b1],
            )
        )
        replacement_index += 1
        opcode_index += 1

    return replacements


def build_line_array_replacement_mapping(
    original_lines: list[str],
    masked_lines: list[str],
    *,
    encoding_name: str = ENCODING_NAME,
) -> dict:
    matcher = SequenceMatcher(a=original_lines, b=masked_lines, autojunk=False)
    replacements: list[LineReplacement] = []
    replacement_index = 0

    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == "equal":
            continue

        original_segment = original_lines[a0:a1]
        masked_segment = masked_lines[b0:b1]

        if opcode == "replace":
            paired = min(len(original_segment), len(masked_segment))
            for offset in range(paired):
                token_replacements = build_token_replacements(
                    original_segment[offset], masked_segment[offset], encoding_name
                )
                replacements.append(
                    LineReplacement(
                        index=replacement_index,
                        opcode="replace",
                        original_line_start=a0 + offset,
                        original_line_end=a0 + offset + 1,
                        masked_line_start=b0 + offset,
                        masked_line_end=b0 + offset + 1,
                        original_lines=[original_segment[offset]],
                        masked_lines=[masked_segment[offset]],
                        token_replacements=token_replacements,
                    )
                )
                replacement_index += 1

            if len(original_segment) > paired:
                for offset in range(paired, len(original_segment)):
                    replacements.append(
                        LineReplacement(
                            index=replacement_index,
                            opcode="delete",
                            original_line_start=a0 + offset,
                            original_line_end=a0 + offset + 1,
                            masked_line_start=b1,
                            masked_line_end=b1,
                            original_lines=[original_segment[offset]],
                            masked_lines=[],
                            token_replacements=[],
                        )
                    )
                    replacement_index += 1

            if len(masked_segment) > paired:
                for offset in range(paired, len(masked_segment)):
                    replacements.append(
                        LineReplacement(
                            index=replacement_index,
                            opcode="insert",
                            original_line_start=a1,
                            original_line_end=a1,
                            masked_line_start=b0 + offset,
                            masked_line_end=b0 + offset + 1,
                            original_lines=[],
                            masked_lines=[masked_segment[offset]],
                            token_replacements=[],
                        )
                    )
                    replacement_index += 1

        elif opcode == "delete":
            for offset in range(len(original_segment)):
                replacements.append(
                    LineReplacement(
                        index=replacement_index,
                        opcode="delete",
                        original_line_start=a0 + offset,
                        original_line_end=a0 + offset + 1,
                        masked_line_start=b0,
                        masked_line_end=b0,
                        original_lines=[original_segment[offset]],
                        masked_lines=[],
                        token_replacements=[],
                    )
                )
                replacement_index += 1

        elif opcode == "insert":
            for offset in range(len(masked_segment)):
                replacements.append(
                    LineReplacement(
                        index=replacement_index,
                        opcode="insert",
                        original_line_start=a0,
                        original_line_end=a0,
                        masked_line_start=b0 + offset,
                        masked_line_end=b0 + offset + 1,
                        original_lines=[],
                        masked_lines=[masked_segment[offset]],
                        token_replacements=[],
                    )
                )
                replacement_index += 1

    return {
        "encoding": encoding_name,
        "original_line_count": len(original_lines),
        "masked_line_count": len(masked_lines),
        "replacement_count": len(replacements),
        "replacements": [
            {
                **asdict(item),
                "token_replacements": [asdict(token_item) for token_item in item.token_replacements],
            }
            for item in replacements
        ],
        "replacement_map": build_flat_replacement_map(replacements),
    }


def build_line_by_line_replacement_mapping(
    original_text: str,
    masked_text: str,
    *,
    encoding_name: str = ENCODING_NAME,
) -> dict:
    original_lines = split_non_empty_lines(original_text)
    masked_lines = split_non_empty_lines(masked_text)
    compared_line_count = min(len(original_lines), len(masked_lines))
    replacements: list[LineReplacement] = []
    replacement_index = 0

    for line_index in range(compared_line_count):
        original_line = original_lines[line_index]
        masked_line = masked_lines[line_index]
        token_replacements = build_token_replacements(original_line, masked_line, encoding_name)

        if not token_replacements and original_line == masked_line:
            continue

        replacements.append(
            LineReplacement(
                index=replacement_index,
                opcode="replace",
                original_line_start=line_index,
                original_line_end=line_index + 1,
                masked_line_start=line_index,
                masked_line_end=line_index + 1,
                original_lines=[original_line],
                masked_lines=[masked_line],
                token_replacements=token_replacements,
            )
        )
        replacement_index += 1

    for line_index in range(compared_line_count, len(original_lines)):
        replacements.append(
            LineReplacement(
                index=replacement_index,
                opcode="delete",
                original_line_start=line_index,
                original_line_end=line_index + 1,
                masked_line_start=compared_line_count,
                masked_line_end=compared_line_count,
                original_lines=[original_lines[line_index]],
                masked_lines=[],
                token_replacements=[],
            )
        )
        replacement_index += 1

    for line_index in range(compared_line_count, len(masked_lines)):
        replacements.append(
            LineReplacement(
                index=replacement_index,
                opcode="insert",
                original_line_start=compared_line_count,
                original_line_end=compared_line_count,
                masked_line_start=line_index,
                masked_line_end=line_index + 1,
                original_lines=[],
                masked_lines=[masked_lines[line_index]],
                token_replacements=[],
            )
        )
        replacement_index += 1

    return {
        "encoding": encoding_name,
        "original_line_count": len(original_lines),
        "masked_line_count": len(masked_lines),
        "compared_line_count": compared_line_count,
        "replacement_count": len(replacements),
        "replacements": [
            {
                **asdict(item),
                "token_replacements": [asdict(token_item) for token_item in item.token_replacements],
            }
            for item in replacements
        ],
        "replacement_map": build_flat_replacement_map(replacements),
    }


def build_json_replacement_mapping(
    stringified_json_a: str,
    stringified_json_b: str,
    *,
    encoding_name: str = ENCODING_NAME,
) -> dict:
    original_lines = parse_json_lines(stringified_json_a)
    masked_lines = parse_json_lines(stringified_json_b)
    return build_line_array_replacement_mapping(
        original_lines,
        masked_lines,
        encoding_name=encoding_name,
    )


def build_flat_replacement_map(replacements: list[LineReplacement]) -> list[dict[str, str]]:
    replacement_map: list[dict[str, str]] = []
    for item in replacements:
        for token_replacement in item.token_replacements:
            replaced_value = normalize_replacement_value(token_replacement.original_text)
            by_value = normalize_replacement_value(token_replacement.masked_text)
            if replaced_value and by_value:
                replacement_map.append(
                    {
                        "replaced": replaced_value,
                        "by": by_value,
                    }
                )
    return replacement_map


def stringify_result(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)