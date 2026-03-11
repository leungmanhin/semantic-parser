import json
import re
import uuid


def stamp_parse_with_uuids(
    stmts: list[str],
    queries: list[str],
) -> tuple[list[str], list[str]]:
    """
    Append a short UUID suffix to every ground identifier (instances and proof
    names) in `stmts` and `queries`. A shared mapping ensures the same base
    name always resolves to the same stamped identifier within one parse, so
    intra-sentence connections are preserved.

    Tokens are stamped iff they match ^[a-z][a-z0-9_]*$ — i.e. snake_case
    identifiers. Everything else (parentheses, ':', '->', variables starting
    with '$', UpperCamelCase predicates/built-ins, numbers, quoted strings) is
    left untouched. Type definitions are excluded by the caller since they
    contain only predicate names and type keywords.
    """
    _GROUND_ID = re.compile(r'^[a-z][a-z0-9_]*$')
    mapping: dict[str, str] = {}

    def stamp_token(tok: str) -> str:
        if _GROUND_ID.match(tok):
            if tok not in mapping:
                mapping[tok] = f"{tok}_{uuid.uuid4().hex[:8]}"
            return mapping[tok]
        return tok

    def stamp_expr(expr: str) -> str:
        tokens = re.findall(r'"[^"]*"|\(|\)|[^\s()]+', expr)
        joined = " ".join(stamp_token(t) for t in tokens)
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
