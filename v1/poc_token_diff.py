from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

import tiktoken


BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_PATH = BASE_DIR / "sample_original.txt"
MASKED_PATH = BASE_DIR / "sample_masked.txt"
OUTPUT_PATH = BASE_DIR / "replacement_mapping.json"
ENCODING_NAME = "cl100k_base"
CHUNK_SIZE_TOKENS = 120


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


def build_replacement_mapping(
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


def load_demo_inputs() -> tuple[str, str]:
    return ORIGINAL_PATH.read_text(encoding="utf-8"), MASKED_PATH.read_text(encoding="utf-8")


def run_demo() -> dict:
    original_text, masked_text = load_demo_inputs()
    result = build_replacement_mapping(original_text, masked_text)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2))