import json
import re
import uuid


def stamp_parse_with_uuids(
    stmts: list[str],
    queries: list[str],
) -> tuple[list[str], list[str]]:
    """
    Append a short UUID suffix to proof names and specific instance identifiers
    in `stmts` and `queries`, leaving generic concept names untouched.

    Under the current naming convention, specific instances always end with a
    numeric suffix (e.g. alice_1, cut_evt_1, apple_group_1) while generic
    concept names do not (e.g. dog, cut_event, fast). A token is stamped iff:
      1. It is a proof name (the snake_case identifier immediately following ':'
         in a PLN expression), OR
      2. It is a snake_case identifier ending with _<digits> (a specific instance)

    Everything else is left untouched: variables ($x), UpperCamelCase predicates
    (IsA, Loves), generic concept names (dog, fast), numbers, quoted strings.

    A shared mapping ensures the same base name always maps to the same stamped
    name within one call, preserving intra-sentence connections.
    """
    _INSTANCE_ID = re.compile(r'^[a-z][a-z0-9_]*_\d+$')
    _SNAKE_ID = re.compile(r'^[a-z][a-z0-9_]*$')
    mapping: dict[str, str] = {}

    def stamp(tok: str) -> str:
        if tok not in mapping:
            mapping[tok] = f"{tok}_{uuid.uuid4().hex[:8]}"
        return mapping[tok]

    def stamp_expr(expr: str) -> str:
        tokens = re.findall(r'"[^"]*"|\(|\)|[^\s()]+', expr)
        result = []
        stamp_next = False

        for tok in tokens:
            if tok in ('(', ')'):
                result.append(tok)
            elif tok == ':':
                result.append(tok)
                stamp_next = True
            elif stamp_next:
                # This token is a proof name — stamp if snake_case, else leave as-is
                # (queries use $prf as proof name, which should not be stamped)
                result.append(stamp(tok) if _SNAKE_ID.match(tok) else tok)
                stamp_next = False
            elif _INSTANCE_ID.match(tok):
                # Specific instance (ends with _<digits>) — stamp it
                result.append(stamp(tok))
            else:
                # Variables, UpperCamelCase predicates, generic concept names,
                # numbers, quoted strings, special tokens — leave untouched
                result.append(tok)

        joined = " ".join(result)
        joined = re.sub(r'\( ', '(', joined)
        joined = re.sub(r' \)', ')', joined)
        return joined

    return (
        [stamp_expr(e) for e in stmts],
        [stamp_expr(e) for e in queries],
    )


def output_to_json_file(json_dict, output_file):
    # print(f"=== Writing to JSON ===\nFile: {output_file}\nContent: {json_dict}\n")
    print(f"=== Writing to JSON ===\nFile: {output_file}\n")
    with open(output_file, "w") as fp:
        json.dump(json_dict, fp, indent=4)
