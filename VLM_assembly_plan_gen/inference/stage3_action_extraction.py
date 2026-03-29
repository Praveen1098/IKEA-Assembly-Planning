"""Stage 3: Robot Action Extraction from assembly manual images.

Design principles (literature-grounded):
  - VLM-as-formalizer: VLM fills action parameters; code composes PDDL (not VLM).
    Source: "Vision Language Models Cannot Plan, but Can They Formalize?" (arXiv 2509.21576)
  - 3-step CoT prompting: visual cues → connection mapping → parameter classification.
    Source: SPAR (2509.13691), NL2Plan (2405.04215)
  - Exhaustive connection enumeration: prompt explicitly asks for ALL connections
    because "VLMs fail to capture exhaustive object relations". (arXiv 2509.21576)
  - Post-VLM completeness check: every new_part must appear in at least one connection.

Taxonomy (Johannsmeier et al. 2025; Ahn et al. 2023; Nemec et al. 2011):
  GRASP    — Positioning: pick up a part  (no spatial params — not needed for PDDL)
  INSERT   — Coupling/Peg-in-hole: translate into socket
  FASTEN   — Coupling+Tooling: tighten with a tool
  PRESS    — Coupling: push together (snap-fit or press-fit)
  REORIENT — Positioning: change part orientation

Inputs (no ground-truth data):
  - tree.json from convert.py → drives step sequence via post-order traversal
  - stage2_output text → step descriptions
  - stage1_output text → part name / ID table
  - data/mask images → step_N_no_seg_numbered.png

Output:
  actions.json with:
    furniture, category, parts, primitives_used, connection_predicates,
    all_connections, steps (list of per-step dicts)
"""

import ast
import json
import os
import re

from config import DATA_DIR, PROMPTS_DIR
from utils import encode_image, ensure_dir


VALID_PRIMITIVES = {"GRASP", "INSERT", "FASTEN", "PRESS", "REORIENT", "PLACE"}
CONNECTION_PRIMITIVES = {"INSERT", "FASTEN", "PRESS"}   # these form PDDL connections


# ---------------------------------------------------------------------------
# Tree parsing and traversal
# ---------------------------------------------------------------------------

def _load_tree(tree_json_path: str):
    """Load and parse tree.json into a nested list.

    convert.py writes: json.dumps(raw_llm_string)
    So the file is a JSON-encoded string whose content is the list literal.
    Decode in two steps: json.load() → string → json.loads() (fallback: ast).
    """
    with open(tree_json_path, "r") as f:
        raw_string = json.load(f)

    raw_string = raw_string.strip()
    try:
        return json.loads(raw_string)
    except (json.JSONDecodeError, TypeError):
        return ast.literal_eval(raw_string)


def _get_leaves(node) -> list[int]:
    if isinstance(node, int):
        return [node]
    leaves = []
    for child in node:
        leaves.extend(_get_leaves(child))
    return leaves


def _post_order_steps(node, steps: list) -> None:
    """Post-order DFS: each inner node = one assembly step.

    Children processed before parents → bottom-up order matching manual pages.
    Each step dict:
      step_idx        : 0-based post-order index (= mask image index)
      all_parts       : all leaf IDs in this subtree
      new_parts       : direct integer children (newly introduced this step)
      subassembly_parts: leaf IDs from child sub-lists (already assembled)
    """
    if isinstance(node, int):
        return

    for child in node:
        _post_order_steps(child, steps)

    new_parts = []
    subassembly_parts = []
    for child in node:
        if isinstance(child, int):
            new_parts.append(child)
        else:
            subassembly_parts.extend(_get_leaves(child))

    steps.append({
        "step_idx": len(steps),
        "all_parts": sorted(_get_leaves(node)),
        "new_parts": sorted(new_parts),
        "subassembly_parts": sorted(subassembly_parts),
    })


def get_assembly_steps(tree) -> list[dict]:
    steps: list[dict] = []
    _post_order_steps(tree, steps)
    return steps


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _load_action_prompt() -> str:
    with open(os.path.join(PROMPTS_DIR, "action_extraction.txt"), "r") as f:
        return f.read()


def _fill_prompt(
    template: str,
    step_idx: int,
    new_parts: list[int],
    subassembly_parts: list[int],
    stage1_parts: str,
    stage2_step: str,
) -> str:
    """Substitute all placeholders using .replace() to avoid conflicts with
    JSON curly braces that appear in the prompt template examples."""
    result = template.replace("{step_idx}", str(step_idx))
    result = result.replace("{new_parts}", str(new_parts))
    result = result.replace("{subassembly_parts}", str(subassembly_parts))
    result = result.replace("{stage1_parts}", stage1_parts)
    result = result.replace("{stage2_step}", stage2_step)
    return result


# ---------------------------------------------------------------------------
# Image and Stage 2 helpers
# ---------------------------------------------------------------------------

def _find_step_image(mask_dir: str, step_idx: int) -> str | None:
    canonical = os.path.join(mask_dir, f"step_{step_idx}_no_seg_numbered.png")
    if os.path.isfile(canonical):
        return encode_image(canonical)
    if os.path.isdir(mask_dir):
        prefix = f"step_{step_idx}_"
        for fname in sorted(os.listdir(mask_dir)):
            if fname.startswith(prefix) and fname.endswith("numbered.png"):
                return encode_image(os.path.join(mask_dir, fname))
    return None


def _parse_stage2_steps(stage2_text: str) -> dict[int, str]:
    """Split Stage 2 plan into per-step strings (1-indexed keys)."""
    steps: dict[int, str] = {}
    current_idx = None
    current_lines: list[str] = []

    for line in stage2_text.splitlines():
        header = re.match(r"#+\s*Step\s+(\d+)", line, re.IGNORECASE)
        if header:
            if current_idx is not None:
                steps[current_idx] = "\n".join(current_lines).strip()
            current_idx = int(header.group(1))
            current_lines = []
        elif current_idx is not None:
            current_lines.append(line)

    if current_idx is not None:
        steps[current_idx] = "\n".join(current_lines).strip()

    return steps


def _clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    return raw


# ---------------------------------------------------------------------------
# Post-VLM validation
# ---------------------------------------------------------------------------

def _validate_connections(
    step_idx: int,
    new_parts: list[int],
    connections_formed: list[dict],
    debug: bool,
) -> list[str]:
    """Warn if any new_part is absent from connections_formed.

    Per VLM→PDDL research (arXiv 2509.21576): VLMs miss exhaustive connections.
    Logging here enables inspection / future refinement loop.
    """
    warnings = []
    connected_parts = set()
    for conn in connections_formed:
        connected_parts.add(conn.get("part1"))
        connected_parts.add(conn.get("part2"))

    for p in new_parts:
        if p not in connected_parts:
            msg = (
                f"[stage3 warning] step {step_idx}: new_part {p} "
                f"does not appear in any connections_formed entry. "
                "VLM may have missed a connection."
            )
            warnings.append(msg)
            if debug:
                print(msg)

    return warnings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_actions(
    furniture_name: str,
    furniture_type: str,
    output_path: str,
    stage2_output: str,
    stage1_output: str,
    args,
    llm,
) -> dict:
    """Extract parameterized robot actions driven by the assembly tree.

    Reads tree.json (written by convert.py) and traverses it bottom-up to
    determine the sequential assembly steps.  No ground-truth data used.

    Returns actions dict with top-level PDDL-ready fields:
      parts, primitives_used, connection_predicates, all_connections, steps
    """
    from llm.utils import invoke_multimodal

    tree_path = os.path.join(output_path, furniture_type, furniture_name, "tree.json")
    if not os.path.isfile(tree_path):
        raise FileNotFoundError(
            f"tree.json not found at {tree_path}. "
            "Run convert_to_tree() before extract_actions()."
        )

    tree = _load_tree(tree_path)
    assembly_steps = get_assembly_steps(tree)

    prompt_template = _load_action_prompt()
    mask_dir = os.path.join(DATA_DIR, "mask", furniture_type, furniture_name)
    stage2_steps = _parse_stage2_steps(stage2_output)
    parts_text = stage1_output.replace("```json", "").replace("```", "").strip()

    result_steps = []

    for step in assembly_steps:
        step_idx = step["step_idx"]
        new_parts = step["new_parts"]
        subassembly_parts = step["subassembly_parts"]

        b64_img = _find_step_image(mask_dir, step_idx)
        if b64_img is None:
            result_steps.append({
                "step_idx": step_idx,
                "new_parts": new_parts,
                "subassembly_parts": subassembly_parts,
                "all_parts": step["all_parts"],
                "stage2_description": "",
                "error": f"No image found for step_{step_idx} in {mask_dir}",
                "actions": [],
                "connections_formed": [],
                "warnings": [],
            })
            continue

        # Stage 2 is 1-indexed; step_idx is 0-indexed
        stage2_key = step_idx + 1
        step_text = stage2_steps.get(stage2_key, stage2_steps.get(step_idx, ""))
        if not step_text:
            if subassembly_parts:
                step_text = (
                    f"Combine new parts {new_parts} with the existing "
                    f"subassembly (parts {subassembly_parts})."
                )
            else:
                step_text = f"Assemble parts {new_parts} together."

        prompt = _fill_prompt(
            prompt_template,
            step_idx=step_idx,
            new_parts=new_parts,
            subassembly_parts=subassembly_parts,
            stage1_parts=parts_text,
            stage2_step=step_text,
        )

        raw_response = invoke_multimodal(llm, prompt, [b64_img], mime_type="image/jpeg")
        cleaned = _clean_json(raw_response)

        try:
            step_actions = json.loads(cleaned)
        except json.JSONDecodeError:
            step_actions = {
                "parse_error": True,
                "raw_response": raw_response,
                "actions": [],
                "connections_formed": [],
            }

        # --- Discard unreliable VLM-generated bookkeeping fields ---
        for key in ("step_id", "parts_involved", "preconditions", "effects"):
            step_actions.pop(key, None)

        # --- Inject authoritative tree-derived fields ---
        step_actions["step_idx"] = step_idx
        step_actions["new_parts"] = new_parts
        step_actions["subassembly_parts"] = subassembly_parts
        step_actions["all_parts"] = step["all_parts"]
        step_actions["stage2_description"] = step_text

        # --- Validate primitive names ---
        for action in step_actions.get("actions", []):
            prim = action.get("primitive", "")
            if prim not in VALID_PRIMITIVES:
                action["primitive_warning"] = (
                    f"Unknown primitive '{prim}'; "
                    f"expected one of {sorted(VALID_PRIMITIVES)}"
                )

        # --- Post-VLM connection completeness check ---
        connections_formed = step_actions.get("connections_formed", [])
        warnings = _validate_connections(
            step_idx, new_parts, connections_formed, args.debug
        )
        step_actions["warnings"] = warnings

        result_steps.append(step_actions)

        if args.debug:
            debug_dir = os.path.join(output_path, furniture_type, furniture_name, "debug")
            ensure_dir(debug_dir)
            with open(os.path.join(debug_dir, f"stage3_prompt_step{step_idx}.txt"), "w") as f:
                f.write(prompt)
            with open(os.path.join(debug_dir, f"stage3_output_step{step_idx}.json"), "w") as f:
                f.write(cleaned)

    # --- Furniture-level aggregation for PDDL generation ---
    all_part_ids: set[int] = set()
    primitives_used: set[str] = set()
    all_connections: list[dict] = []

    for step in result_steps:
        all_part_ids.update(step.get("all_parts", []))
        for action in step.get("actions", []):
            prim = action.get("primitive", "")
            if prim in VALID_PRIMITIVES:
                primitives_used.add(prim)
        for conn in step.get("connections_formed", []):
            if isinstance(conn, dict) and conn not in all_connections:
                all_connections.append(conn)

    connection_predicates = sorted({
        conn["predicate"]
        for conn in all_connections
        if "predicate" in conn
    })

    actions_result = {
        "furniture": furniture_name,
        "category": furniture_type,
        "parts": sorted(all_part_ids),
        "primitives_used": sorted(primitives_used),
        "connection_predicates": connection_predicates,
        "all_connections": all_connections,
        "steps": result_steps,
    }

    item_out_dir = os.path.join(output_path, furniture_type, furniture_name)
    ensure_dir(item_out_dir)
    actions_path = os.path.join(item_out_dir, "actions.json")
    with open(actions_path, "w") as f:
        json.dump(actions_result, f, indent=2)

    if args.debug:
        print(f"Saved actions to {actions_path}")

    return actions_result
