"""
Role-Filler Distribution Clustering — Paper-Faithful Approach  (FUSE-NF §4.3.2)
=================================================================================
For every predicate-slot in a corpus of SENF/PLN graphs, collects the actual
filler names, embeds them in a vector space, and clusters slots whose filler
populations are distributionally similar.

This follows the paper directly:

    "For every predicate-slot (e.g. go_to.Agent), we collect the set of fillers
     across the corpus and embed them in a vector space.  Clustering these
     embeddings reveals when two slots share indistinguishable distributions of
     fillers, indicating they fulfill the same semantic role and can be merged."

Filler names are made human-readable by stripping instance suffixes
(``maria_1`` → ``"maria"``) and replacing underscores with spaces
(``skeletal_muscle`` → ``"skeletal muscle"``).

The output is simply a list of slot clusters — groups of predicate-slots whose
filler populations are distributionally similar.  No further categorisation
(ND vs non-ND, partial vs full equivalence) is imposed; that is left to
downstream consumers or an LLM-based consolidation step.

Event-conditioned slot keys (``drive_event.Agent.arg1`` instead of plain
``Agent.arg1``) are still generated to prevent generic bridge predicates like
``Subject`` from pooling fillers across unrelated event types.
"""

import re
import numpy as np
from collections import Counter, defaultdict
from typing import Optional

from pydantic import BaseModel

from llm import get_embedding, to_openrouter
from built_in import built_in_ops
from prompts import base_instructions

# Reuse S-expression parser, event detection, and clustering from the
# type-label approach.
from role_filler_clustering import (
    _parse_sexp,
    _get_core,
    _is_event_type,
    _event_ctx_of,
    cluster_slots,
)


# 
# Filler name conversion
# 

_INSTANCE_SUFFIX_RE = re.compile(r'_\d+$')


def _filler_to_str(arg) -> str | None:
    """
    Convert a raw S-expression argument into an embeddable filler string.

    - String starting with ``$`` (variable) → ``None`` (skip).
    - Other string → strip trailing ``_\\d+`` suffix, replace ``_`` with space.
    - List whose head is a non-built-in predicate → return head as-is.
    - List whose head is a built-in op → ``None`` (structural, not a filler).
    - Anything else → ``None``.
    """
    if isinstance(arg, str):
        if arg.startswith('$'):
            return None
        return _INSTANCE_SUFFIX_RE.sub('', arg).replace('_', ' ')
    if isinstance(arg, list) and arg:
        head = arg[0]
        if isinstance(head, str) and head not in built_in_ops:
            return head
        return None
    return None


# 
# Phase 1 – Collect raw filler names per predicate-slot
# 

def _collect_slot_fillers(
    expr,
    slot_fillers: dict[str, set[str]],
    var_types: dict[str, set[str]] | None = None,
) -> None:
    """
    Recursively walk ``expr`` and, for every multi-argument domain predicate
    application ``(P a0 a1 … aN)`` (N ≥ 1), record human-readable filler
    strings per slot key.

    Variables (``$x``) are **skipped** as fillers — only ground constants and
    sub-expression heads contribute.  However, ``var_types`` is still
    maintained for Neo-Davidsonian event-type detection on arg0.

    Slot key format:

    - ``"P.argI"`` — plain, when arg0 is not event-typed.
    - ``"EventType.P.argI"`` — event-conditioned, when arg0 is typed by a
      single ND event predicate (arg0 itself is skipped).
    """
    if var_types is None:
        var_types = {}
    if not isinstance(expr, list) or not expr:
        return
    head = expr[0]
    if not isinstance(head, str):
        return

    # === And: build local type context (for ND detection) 
    #   Captures types for BOTH variables ($e) and ground instances
    #   (drive_evt_1) so that event-conditioned slot keys are generated
    #   correctly in either case.
    if head == 'And':
        atoms = expr[1:]
        local_var_types: dict[str, set[str]] = {k: set(v) for k, v in var_types.items()}
        for atom in atoms:
            if not isinstance(atom, list) or not atom:
                continue
            p = atom[0]
            if not isinstance(p, str):
                continue
            if p == 'IsA' and len(atom) == 3:
                subj, concept = atom[1], atom[2]
                if isinstance(subj, str) and isinstance(concept, str):
                    local_var_types.setdefault(subj, set()).add(concept)
            elif p == 'HasAttribute' and len(atom) == 3:
                subj, attr = atom[1], atom[2]
                if isinstance(subj, str) and isinstance(attr, str):
                    local_var_types.setdefault(subj, set()).add(attr)
            elif p not in built_in_ops and len(atom) == 2:
                a = atom[1]
                if isinstance(a, str):
                    local_var_types.setdefault(a, set()).add(p)
        for atom in atoms:
            _collect_slot_fillers(atom, slot_fillers, local_var_types)
        return

    # === Implication: share antecedent type context with consequent 
    if head == 'Implication' and len(expr) >= 3:
        antecedent, consequent = expr[1], expr[2]
        shared_var_types: dict[str, set[str]] = {k: set(v) for k, v in var_types.items()}
        if isinstance(antecedent, list) and antecedent and antecedent[0] == 'And':
            for atom in antecedent[1:]:
                if not isinstance(atom, list) or not atom:
                    continue
                p = atom[0]
                if not isinstance(p, str):
                    continue
                if p == 'IsA' and len(atom) == 3:
                    subj, concept = atom[1], atom[2]
                    if isinstance(subj, str) and isinstance(concept, str):
                        shared_var_types.setdefault(subj, set()).add(concept)
                elif p == 'HasAttribute' and len(atom) == 3:
                    subj, attr = atom[1], atom[2]
                    if isinstance(subj, str) and isinstance(attr, str):
                        shared_var_types.setdefault(subj, set()).add(attr)
                elif p not in built_in_ops and len(atom) == 2:
                    a = atom[1]
                    if isinstance(a, str):
                        shared_var_types.setdefault(a, set()).add(p)
        _collect_slot_fillers(antecedent, slot_fillers, var_types)
        _collect_slot_fillers(consequent, slot_fillers, shared_var_types)
        return

    # === Other built-in connectives 
    if head in built_in_ops:
        for child in expr[1:]:
            _collect_slot_fillers(child, slot_fillers, var_types)
        return

    # === Domain predicate application 
    args = expr[1:]
    if len(args) > 1:
        # Detect ND event context from arg0 (works for both variables
        # like $e and ground instances like drive_evt_1).
        a0_types: set[str] = set()
        if isinstance(args[0], str):
            a0_types = var_types.get(args[0], set())
        event_ctx = _event_ctx_of(a0_types)

        for i, arg in enumerate(args):
            if event_ctx is not None:
                if i == 0:
                    continue  # skip: event type already in key prefix
                slot_key = f"{event_ctx}.{head}.arg{i}"
            else:
                slot_key = f"{head}.arg{i}"
            filler = _filler_to_str(arg)
            if filler is not None:
                slot_fillers.setdefault(slot_key, set()).add(filler)

    # Recurse into list-valued arguments.
    for arg in args:
        if isinstance(arg, list):
            _collect_slot_fillers(arg, slot_fillers, var_types)


def extract_slot_fillers(
    corpus: list[dict],
    min_fillers: int = 2,
) -> dict[str, list[str]]:
    """
    Single pass over the corpus.  For every predicate-slot, collects the set
    of unique human-readable filler strings.

    Parameters
    ----------
    corpus      : list of PLN parse dicts (each must have ``"stmts"``).
    min_fillers : minimum number of distinct fillers to keep a slot.

    Returns
    -------
    ``{"Pred.argN" | "EventType.Pred.argN": [filler_str, ...]}``
    Fillers are sorted and deduplicated.
    """
    slot_fillers: dict[str, set[str]] = defaultdict(set)
    for item in corpus:
        for stmt in item.get('stmts', []):
            parsed = _parse_sexp(stmt)
            core = _get_core(parsed)
            _collect_slot_fillers(core, slot_fillers)
    return {
        slot: sorted(fillers)
        for slot, fillers in slot_fillers.items()
        if len(fillers) >= min_fillers
    }


# 
# Phase 2 – Embed filler names → per-slot centroid vectors
# 

def embed_slot_fillers(
    slot_fillers: dict[str, list[str]],
    embedding_cache: Optional[dict[str, np.ndarray]] = None,
    on_new_embedding: Optional[callable] = None,
    verbose: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Embeds each unique filler string (e.g. ``"maria"``, ``"camera"``) and
    computes each slot's centroid as the unweighted mean of its constituent
    filler embeddings, normalised to unit length.

    Parameters
    ----------
    slot_fillers      : output of :func:`extract_slot_fillers`.
    embedding_cache   : optional ``{filler_str: vector}`` dict; updated
                        in-place and returned for reuse across batches.
    on_new_embedding  : optional callback after each new embedding, receiving
                        the updated cache.
    verbose           : print progress information.

    Returns
    -------
    slot_centroids  : ``{"Pred.argN": unit_vector}``
    embedding_cache : updated cache
    """
    if embedding_cache is None:
        embedding_cache = {}

    all_fillers = {f for fillers in slot_fillers.values() for f in fillers}
    new_fillers = all_fillers - embedding_cache.keys()
    if verbose and new_fillers:
        print(f"  Fetching embeddings for {len(new_fillers)} filler name(s) …")
    for filler in sorted(new_fillers):
        if verbose:
            print(f"    embed: {filler!r}")
        embedding_cache[filler] = get_embedding(filler)
        if on_new_embedding is not None:
            on_new_embedding(embedding_cache)

    dim = next(iter(embedding_cache.values())).shape[0]
    slot_centroids: dict[str, np.ndarray] = {}
    for slot, fillers in slot_fillers.items():
        if not fillers:
            continue
        centroid = np.zeros(dim)
        for f in fillers:
            centroid += embedding_cache[f]
        centroid /= len(fillers)
        norm = np.linalg.norm(centroid)
        slot_centroids[slot] = centroid / norm if norm > 0 else centroid

    return slot_centroids, embedding_cache


# 
# Merge suggestions — within-predicate slot merges only
# 

_ND_SLOT_RE = re.compile(r'^([^.]+)\.([^.]+)\.(arg\d+)$')
_PLAIN_SLOT_RE = re.compile(r'^([^.]+)\.(arg\d+)$')


def _slot_predicate(slot: str) -> str | None:
    """
    Extract the predicate (or event type) that a slot belongs to.

    - ``"drive_event.Agent.arg1"`` → ``"drive_event"``
    - ``"Eat.arg0"``              → ``"Eat"``
    """
    m = _ND_SLOT_RE.match(slot)
    if m and _is_event_type(m.group(1)):
        return m.group(1)
    m = _PLAIN_SLOT_RE.match(slot)
    if m:
        return m.group(1)
    return None


def _is_nd_slot(slot: str) -> bool:
    """Return True if *slot* is an event-conditioned ND key."""
    m = _ND_SLOT_RE.match(slot)
    return bool(m) and _is_event_type(m.group(1))


def suggest_merges(
    clusters: list[list[str]],
    event_conditioned_only: bool = True,
) -> list[list[str]]:
    """
    Return within-predicate merge candidates following the paper's intent
    (§4.3.2): only slots belonging to the **same** predicate or event type
    that cluster together are merge candidates.

    For example, ``drive_event.Agent.arg1`` and ``drive_event.Accompany.arg1``
    clustering together is a valid within-predicate merge (same event type).
    But ``CenterOf.arg1`` and ``In.arg1`` is not (different predicates).

    Parameters
    ----------
    clusters               : output of :func:`cluster_slots`.
    event_conditioned_only : when ``True`` (default), only event-conditioned
                             ND slots (``event_type.Role.argN``) are
                             considered.  Plain predicate slots like
                             ``Cause.arg0`` are skipped because merging
                             positional args of a plain predicate is rarely
                             meaningful.  Set to ``False`` to include plain
                             predicate within-slot merges as well.
    """
    results: list[list[str]] = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        # Group slots in this cluster by their predicate/event-type.
        by_pred: dict[str, list[str]] = defaultdict(list)
        for slot in cluster:
            if event_conditioned_only and not _is_nd_slot(slot):
                continue
            pred = _slot_predicate(slot)
            if pred is not None:
                by_pred[pred].append(slot)
        # Only within-predicate groups with 2+ slots are merge candidates.
        for pred, slots in by_pred.items():
            if len(slots) > 1:
                results.append(sorted(slots))
    return results


# 
# LLM-based consolidation rule generation
# 

class _ConsolidationResult(BaseModel):
    type_defs: list[str]
    source_groups: list[str]
    rules: list[str]
    rules_nl: list[str]


_consolidation_system_prompt = f"""
<guidelines>
{base_instructions}
</guidelines>

You are analyzing the output of a role-filler distribution clustering pipeline applied to
a corpus of PLN knowledge graphs.  The pipeline collects, for every predicate-argument slot, the set of
actual filler names across the corpus, embeds them, and clusters slots whose filler populations are
distributionally similar.

Each candidate group contains **within-predicate** slot merges: multiple role slots belonging to the
**same** predicate or event type that attract indistinguishable filler populations.  For example,
`drive_event.Agent.arg1` and `drive_event.Accompany.arg1` both attract person names — they serve
the same semantic role within `drive_event` and can be collapsed into a single canonical role.

Your task is to evaluate each candidate group and — where semantically justified — generate PLN
consolidation rules that map the redundant role slots to a single canonical role predicate.

For event-conditioned slots (e.g. `drive_event.Agent.arg1` and `drive_event.Accompany.arg1`), generate
one Implication rule per slot, each mapping to a canonical combined role:
    `(: agent_to_participant (Implication (And (IsA $e drive_event) (Agent $e $x)) (Participant $e $x)) (STV 0.95 0.9))`
    `(: accompany_to_participant (Implication (And (IsA $e drive_event) (Accompany $e $x)) (Participant $e $x)) (STV 0.95 0.9))`

For plain predicate slots (e.g. `Eat.arg0` and `Eat.arg1` — rare, since different arg positions of the
same predicate rarely attract similar fillers), generate rules only if the merge is genuinely meaningful.

For each group:
- Inspect the filler names to judge whether the similarity is semantically meaningful.
- Choose a good canonical role name that describes the merged concept (e.g. Participant, Experiencer).
- If a suggestion looks coincidental or spurious, skip it — do NOT invent rules just to have something.
- All rules must be syntactically valid PLN expressions wrapped as `(: <proof_name> <body> (STV <s> <c>))`.
- Provide a type definition for every new predicate you introduce.

Return four parallel flat lists (all lists must have the same length, except type_defs):
- type_defs    : type definitions for any new predicates introduced in the rules
- source_groups: the label of the candidate group each rule addresses (e.g. "1", "2", "3"),
                 one entry per rule, in the same order as `rules` and `rules_nl`
- rules        : PLN Implication expressions, one per rule, each wrapped as
                 `(: <proof_name> <body> (STV <s> <c>))`
- rules_nl     : one-line English descriptions, one per rule
""".strip()


def _format_merge_evidence(
    merge_candidates: list[list[str]],
    slot_fillers: dict[str, list[str]],
    top_k: int = 10,
) -> str:
    """
    Format merge candidates and their filler lists into a human-readable
    string for the LLM prompt.
    """
    lines: list[str] = []
    for i, group in enumerate(merge_candidates, 1):
        lines.append(f"[{i}]")
        for slot in group:
            fillers = slot_fillers.get(slot, [])
            shown = fillers[:top_k]
            suffix = f" … and {len(fillers) - top_k} more" if len(fillers) > top_k else ""
            lines.append(f"  {slot}  →  {shown}{suffix}")
        lines.append("")
    return "\n".join(lines)


def generate_consolidation_rules(
    merge_candidates: list[list[str]],
    slot_fillers: dict[str, list[str]],
    model: str = "gpt-5.4",
    effort: str = "high",
    verbose: bool = True,
) -> dict:
    """
    Ask an LLM to evaluate merge suggestions and generate PLN consolidation
    rules.

    Parameters
    ----------
    merge_candidates : non-singleton clusters from :func:`suggest_merges`.
    slot_fillers     : from :func:`extract_slot_fillers`.
    model            : LLM model identifier.
    effort           : reasoning effort level.
    verbose          : print progress information.

    Returns
    -------
    dict with ``'type_defs'`` and ``'rules'`` (list of dicts, each with
    ``'source_group'``, ``'rule'``, ``'rule_nl'``, ``'candidates'``).
    """
    if not merge_candidates:
        if verbose:
            print("  No merge candidates — skipping LLM consolidation step.")
        return {"type_defs": [], "rules": []}

    group_index: dict[str, list[str]] = {}
    for i, group in enumerate(merge_candidates, 1):
        group_index[str(i)] = group

    evidence = _format_merge_evidence(merge_candidates, slot_fillers)

    if verbose:
        print(f"  {len(merge_candidates)} merge candidate group(s):")
        for line in evidence.splitlines():
            print(f"    {line}")
        print(f"\n  Calling LLM (model={model!r}, effort={effort!r}) …")

    history = [{"role": "system", "content": _consolidation_system_prompt}]
    prompt = (
        "Below are the merge candidate groups from role-filler distribution clustering.\n"
        "For each group, the filler lists show which entities fill that argument position "
        "across the corpus.\n\n"
        f"{evidence}\n"
        "Please evaluate each group and generate PLN consolidation rules where semantically "
        "justified.  Set source_group to the group number (e.g. '1', '2', '3').  "
        "Skip any group that looks coincidental or spurious."
    )

    raw = to_openrouter(
        prompt,
        model=model,
        effort=effort,
        history=history,
        output_format=_ConsolidationResult,
    )

    source_groups = raw.get("source_groups", [])
    rules         = raw.get("rules", [])
    rules_nl      = raw.get("rules_nl", [])
    enriched_rules = [
        {
            "source_group": label,
            "rule":         rule,
            "rule_nl":      nl,
            "candidates":   group_index.get(label, []),
        }
        for label, rule, nl in zip(source_groups, rules, rules_nl)
    ]

    if verbose:
        print(f"  LLM generated {len(enriched_rules)} consolidation rule(s).")

    return {"type_defs": raw.get("type_defs", []), "rules": enriched_rules}


# 
# End-to-end entry point
# 

def run_role_filler_clustering_2(
    corpus: list[dict],
    similarity_threshold: float = 0.85,
    min_fillers: int = 2,
    event_conditioned_only: bool = True,
    embedding_cache: Optional[dict[str, np.ndarray]] = None,
    on_new_embedding: Optional[callable] = None,
    verbose: bool = True,
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Paper-faithful pipeline: corpus → filler extraction → filler-name
    embedding → slot clustering → merge suggestions.

    Parameters
    ----------
    corpus                 : list of PLN parse dicts.
    similarity_threshold   : cosine similarity cutoff for merging slots.
    min_fillers            : minimum distinct fillers per slot.
    event_conditioned_only : only consider ND event-conditioned slots for
                             merge suggestions (default ``True``).
    embedding_cache        : reusable ``{filler_str: vector}`` cache.
    on_new_embedding       : callback after each new embedding.
    verbose                : print progress information.

    Returns
    -------
    all_clusters     : full slot partition (including singletons).
    merge_candidates : within-predicate slot groups with similar fillers.
    """
    if verbose:
        print(f"[1/3] Extracting slot fillers from {len(corpus)} corpus items …")
    slot_fillers = extract_slot_fillers(corpus, min_fillers=min_fillers)
    if verbose:
        print(f"      {len(slot_fillers)} qualifying predicate-slot(s) found.")
        for slot, fillers in list(slot_fillers.items())[:5]:
            print(f"        {slot}: {fillers[:6]}")

    if not slot_fillers:
        if verbose:
            print("      No slots with enough fillers — nothing to cluster.")
        return [], []

    if verbose:
        print(f"\n[2/3] Embedding filler names …")
    slot_centroids, embedding_cache = embed_slot_fillers(
        slot_fillers, embedding_cache=embedding_cache,
        on_new_embedding=on_new_embedding, verbose=verbose,
    )

    if verbose:
        print(f"\n[3/3] Clustering {len(slot_centroids)} slot(s) "
              f"(threshold={similarity_threshold}) …")
    all_clusters = cluster_slots(slot_centroids, similarity_threshold)
    merge_candidates = suggest_merges(all_clusters, event_conditioned_only=event_conditioned_only)

    if verbose:
        print(f"\n=== Results ")
        print(f"  {len(all_clusters)} cluster(s) total.")
        print(f"  {len(merge_candidates)} merge candidate group(s):")
        for i, cluster in enumerate(merge_candidates, 1):
            print(f"  [{i}] {cluster}")

    return all_clusters, merge_candidates
