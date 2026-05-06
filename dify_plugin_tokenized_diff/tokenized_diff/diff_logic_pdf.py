from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from diff_logic import ENCODING_NAME
from diff_logic_gitlike import build_gitlike_line_replacement_mapping_for_lines


SECTION_HEADERS = {
    "Call to Order",
    "Key Discussions",
    "Action Items",
    "Adjournment",
}
METADATA_PREFIXES = (
    "Date:",
    "Time:",
    "Location:",
    "Absent:",
    "Recorder:",
    "Word Count:",
)
MERGEABLE_METADATA_PREFIXES = (
    "Attendees:",
)
SECTION_SUBHEADING_PATTERN = re.compile(r"^.+\(Led by .+\)$")
ACTION_ITEM_PATTERN = re.compile(r"^.+\(Owner: .+\)\.?$")


@dataclass
class NormalizedPdfLine:
    index: int
    kind: str
    text: str
    source_line_numbers: list[int]
    source_filtered_indexes: list[int]


def classify_pdf_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped in SECTION_HEADERS:
        return "section_header"
    if any(stripped.startswith(prefix) for prefix in MERGEABLE_METADATA_PREFIXES):
        return None
    if any(stripped.startswith(prefix) for prefix in METADATA_PREFIXES):
        return "metadata"
    if SECTION_SUBHEADING_PATTERN.match(stripped):
        return "section_subheading"
    if ACTION_ITEM_PATTERN.match(stripped):
        return "action_item"
    return None


def normalize_pdf_extracted_lines(text: str) -> tuple[list[NormalizedPdfLine], int]:
    filtered_physical_lines: list[tuple[int, int, str]] = []
    for source_line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if stripped:
            filtered_physical_lines.append(
                (len(filtered_physical_lines), source_line_number, stripped)
            )

    normalized_lines: list[NormalizedPdfLine] = []
    body_parts: list[str] = []
    body_source_line_numbers: list[int] = []
    body_source_filtered_indexes: list[int] = []

    def flush_body() -> None:
        if not body_parts:
            return
        normalized_lines.append(
            NormalizedPdfLine(
                index=len(normalized_lines),
                kind="body",
                text=" ".join(body_parts),
                source_line_numbers=body_source_line_numbers.copy(),
                source_filtered_indexes=body_source_filtered_indexes.copy(),
            )
        )
        body_parts.clear()
        body_source_line_numbers.clear()
        body_source_filtered_indexes.clear()

    for filtered_index, source_line_number, stripped in filtered_physical_lines:
        line_kind = classify_pdf_line(stripped)
        if line_kind is not None:
            flush_body()
            normalized_lines.append(
                NormalizedPdfLine(
                    index=len(normalized_lines),
                    kind=line_kind,
                    text=stripped,
                    source_line_numbers=[source_line_number],
                    source_filtered_indexes=[filtered_index],
                )
            )
            continue

        body_parts.append(stripped)
        body_source_line_numbers.append(source_line_number)
        body_source_filtered_indexes.append(filtered_index)

    flush_body()
    return normalized_lines, len(filtered_physical_lines)


def build_pdf_extract_gitlike_replacement_mapping(
    original_text: str,
    masked_text: str,
    *,
    encoding_name: str = ENCODING_NAME,
) -> dict:
    original_normalized_lines, original_filtered_line_count = normalize_pdf_extracted_lines(
        original_text
    )
    masked_normalized_lines, masked_filtered_line_count = normalize_pdf_extracted_lines(masked_text)

    result = build_gitlike_line_replacement_mapping_for_lines(
        [item.text for item in original_normalized_lines],
        [item.text for item in masked_normalized_lines],
        encoding_name=encoding_name,
        line_filter="drop_blank_trimmed_lines_then_merge_pdf_body_lines",
        normalization={
            "normalization_strategy": "preserve_standalone_headers_metadata_and_action_items_merge_consecutive_body_lines",
            "original_filtered_line_count": original_filtered_line_count,
            "masked_filtered_line_count": masked_filtered_line_count,
            "original_normalized_line_count": len(original_normalized_lines),
            "masked_normalized_line_count": len(masked_normalized_lines),
            "original_normalization_map": [asdict(item) for item in original_normalized_lines],
            "masked_normalization_map": [asdict(item) for item in masked_normalized_lines],
        },
    )
    return result