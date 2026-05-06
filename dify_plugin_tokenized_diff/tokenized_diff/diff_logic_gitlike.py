from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

from diff_logic import (
    ENCODING_NAME,
    LineReplacement,
    build_flat_replacement_map,
    build_token_replacements,
    split_non_empty_lines,
)


LINE_PAIR_SIMILARITY_THRESHOLD = 0.35
UNPAIRABLE_LINE_COST = 3.0
INSERT_DELETE_COST = 1.0


@dataclass
class DiffHunk:
    index: int
    original_line_start: int
    original_line_end: int
    masked_line_start: int
    masked_line_end: int
    operations: list[LineReplacement]


def line_similarity(original_line: str, masked_line: str) -> float:
    original_normalized = original_line.strip()
    masked_normalized = masked_line.strip()
    if not original_normalized and not masked_normalized:
        return 1.0
    return SequenceMatcher(
        a=original_normalized,
        b=masked_normalized,
        autojunk=False,
    ).ratio()


def _build_delete_operation(
    original_line: str,
    original_index: int,
    masked_anchor: int,
    replacement_index: int,
) -> LineReplacement:
    return LineReplacement(
        index=replacement_index,
        opcode="delete",
        original_line_start=original_index,
        original_line_end=original_index + 1,
        masked_line_start=masked_anchor,
        masked_line_end=masked_anchor,
        original_lines=[original_line],
        masked_lines=[],
        token_replacements=[],
    )


def _build_insert_operation(
    masked_line: str,
    original_anchor: int,
    masked_index: int,
    replacement_index: int,
) -> LineReplacement:
    return LineReplacement(
        index=replacement_index,
        opcode="insert",
        original_line_start=original_anchor,
        original_line_end=original_anchor,
        masked_line_start=masked_index,
        masked_line_end=masked_index + 1,
        original_lines=[],
        masked_lines=[masked_line],
        token_replacements=[],
    )


def _build_replace_operation(
    original_line: str,
    masked_line: str,
    original_index: int,
    masked_index: int,
    replacement_index: int,
    encoding_name: str,
) -> LineReplacement | None:
    token_replacements = build_token_replacements(original_line, masked_line, encoding_name)
    if not token_replacements and original_line == masked_line:
        return None

    return LineReplacement(
        index=replacement_index,
        opcode="replace",
        original_line_start=original_index,
        original_line_end=original_index + 1,
        masked_line_start=masked_index,
        masked_line_end=masked_index + 1,
        original_lines=[original_line],
        masked_lines=[masked_line],
        token_replacements=token_replacements,
    )


def refine_replace_block(
    original_segment: list[str],
    masked_segment: list[str],
    *,
    original_start: int,
    masked_start: int,
    encoding_name: str = ENCODING_NAME,
) -> list[LineReplacement]:
    original_count = len(original_segment)
    masked_count = len(masked_segment)
    costs = [[0.0] * (masked_count + 1) for _ in range(original_count + 1)]
    steps: list[list[str | None]] = [[None] * (masked_count + 1) for _ in range(original_count + 1)]

    for original_index in range(1, original_count + 1):
        costs[original_index][0] = original_index * INSERT_DELETE_COST
        steps[original_index][0] = "delete"

    for masked_index in range(1, masked_count + 1):
        costs[0][masked_index] = masked_index * INSERT_DELETE_COST
        steps[0][masked_index] = "insert"

    for original_index in range(1, original_count + 1):
        for masked_index in range(1, masked_count + 1):
            similarity = line_similarity(
                original_segment[original_index - 1],
                masked_segment[masked_index - 1],
            )
            pair_cost = (
                1.0 - similarity
                if similarity >= LINE_PAIR_SIMILARITY_THRESHOLD
                else UNPAIRABLE_LINE_COST
            )

            replace_total = costs[original_index - 1][masked_index - 1] + pair_cost
            delete_total = costs[original_index - 1][masked_index] + INSERT_DELETE_COST
            insert_total = costs[original_index][masked_index - 1] + INSERT_DELETE_COST

            best_total = replace_total
            best_step = "pair"

            if delete_total < best_total:
                best_total = delete_total
                best_step = "delete"

            if insert_total < best_total:
                best_total = insert_total
                best_step = "insert"

            costs[original_index][masked_index] = best_total
            steps[original_index][masked_index] = best_step

    aligned_steps: list[tuple[str, int | None, int | None]] = []
    original_index = original_count
    masked_index = masked_count
    while original_index > 0 or masked_index > 0:
        step = steps[original_index][masked_index]
        if step == "pair":
            aligned_steps.append((step, original_index - 1, masked_index - 1))
            original_index -= 1
            masked_index -= 1
            continue
        if step == "delete":
            aligned_steps.append((step, original_index - 1, None))
            original_index -= 1
            continue
        if step == "insert":
            aligned_steps.append((step, None, masked_index - 1))
            masked_index -= 1
            continue
        raise ValueError("Failed to reconstruct replace-block alignment.")

    aligned_steps.reverse()
    replacements: list[LineReplacement] = []
    current_original_index = original_start
    current_masked_index = masked_start

    for step, original_offset, masked_offset in aligned_steps:
        replacement_index = len(replacements)
        if step == "pair":
            assert original_offset is not None
            assert masked_offset is not None
            operation = _build_replace_operation(
                original_segment[original_offset],
                masked_segment[masked_offset],
                current_original_index,
                current_masked_index,
                replacement_index,
                encoding_name,
            )
            if operation is not None:
                replacements.append(operation)
            current_original_index += 1
            current_masked_index += 1
            continue

        if step == "delete":
            assert original_offset is not None
            replacements.append(
                _build_delete_operation(
                    original_segment[original_offset],
                    current_original_index,
                    current_masked_index,
                    replacement_index,
                )
            )
            current_original_index += 1
            continue

        assert masked_offset is not None
        replacements.append(
            _build_insert_operation(
                masked_segment[masked_offset],
                current_original_index,
                current_masked_index,
                replacement_index,
            )
        )
        current_masked_index += 1

    return replacements


def build_gitlike_line_replacement_mapping(
    original_text: str,
    masked_text: str,
    *,
    encoding_name: str = ENCODING_NAME,
) -> dict:
    original_lines = split_non_empty_lines(original_text)
    masked_lines = split_non_empty_lines(masked_text)
    return build_gitlike_line_replacement_mapping_for_lines(
        original_lines,
        masked_lines,
        encoding_name=encoding_name,
        line_filter="drop_blank_trimmed_lines",
    )


def build_gitlike_line_replacement_mapping_for_lines(
    original_lines: list[str],
    masked_lines: list[str],
    *,
    encoding_name: str = ENCODING_NAME,
    line_filter: str = "custom_lines",
    normalization: dict | None = None,
) -> dict:
    matcher = SequenceMatcher(a=original_lines, b=masked_lines, autojunk=False)
    replacements: list[LineReplacement] = []
    hunks: list[DiffHunk] = []

    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == "equal":
            continue

        original_segment = original_lines[a0:a1]
        masked_segment = masked_lines[b0:b1]
        hunk_operations: list[LineReplacement] = []

        if opcode == "replace":
            refined_operations = refine_replace_block(
                original_segment,
                masked_segment,
                original_start=a0,
                masked_start=b0,
                encoding_name=encoding_name,
            )
            for operation in refined_operations:
                hunk_operations.append(
                    LineReplacement(
                        index=len(replacements),
                        opcode=operation.opcode,
                        original_line_start=operation.original_line_start,
                        original_line_end=operation.original_line_end,
                        masked_line_start=operation.masked_line_start,
                        masked_line_end=operation.masked_line_end,
                        original_lines=operation.original_lines,
                        masked_lines=operation.masked_lines,
                        token_replacements=operation.token_replacements,
                    )
                )
                replacements.append(hunk_operations[-1])

        elif opcode == "delete":
            for offset, original_line in enumerate(original_segment):
                operation = _build_delete_operation(
                    original_line,
                    a0 + offset,
                    b0,
                    len(replacements),
                )
                hunk_operations.append(operation)
                replacements.append(operation)

        elif opcode == "insert":
            for offset, masked_line in enumerate(masked_segment):
                operation = _build_insert_operation(
                    masked_line,
                    a0,
                    b0 + offset,
                    len(replacements),
                )
                hunk_operations.append(operation)
                replacements.append(operation)

        if hunk_operations:
            hunks.append(
                DiffHunk(
                    index=len(hunks),
                    original_line_start=min(operation.original_line_start for operation in hunk_operations),
                    original_line_end=max(operation.original_line_end for operation in hunk_operations),
                    masked_line_start=min(operation.masked_line_start for operation in hunk_operations),
                    masked_line_end=max(operation.masked_line_end for operation in hunk_operations),
                    operations=hunk_operations,
                )
            )

    return {
        "encoding": encoding_name,
        "line_filter": line_filter,
        "alignment_strategy": "line_diff_then_refined_replace_blocks_then_token_diff",
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
        "hunk_count": len(hunks),
        "hunks": [
            {
                **asdict(hunk),
                "operations": [
                    {
                        **asdict(operation),
                        "token_replacements": [
                            asdict(token_item) for token_item in operation.token_replacements
                        ],
                    }
                    for operation in hunk.operations
                ],
            }
            for hunk in hunks
        ],
        "replacement_map": build_flat_replacement_map(replacements),
        **(normalization or {}),
    }