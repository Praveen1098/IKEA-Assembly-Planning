"""Stage 3: Robot Action Extraction from assembly manual images.

Design principles (literature-grounded):
  - VLM-as-formalizer: VLM identifies connections; code composes the BT (not VLM).
    Source: "Vision Language Models Cannot Plan, but Can They Formalize?" (arXiv 2509.21576)
  - 3-step CoT prompting: visual cues → connection mapping → parameter extraction.
    Source: SPAR (2509.13691), NL2Plan (2405.04215)
  - Exhaustive connection enumeration: prompt explicitly asks for ALL connections
    because "VLMs fail to capture exhaustive object relations". (arXiv 2509.21576)
  - Iterative refinement loop: if any new_part lacks a connection after the initial
    VLM call, re-prompt with a targeted follow-up naming missing parts explicitly.
    Source: arXiv 2409.09435 — "BT Generation using LLMs with Human Instructions and Feedback"
  - VLM-evaluated condition nodes: visual_description on each connection enables
    a robot VLM to verify assembly state from camera at runtime (arXiv 2501.03968).
  - Post-VLM completeness check: every new_part must appear in at least one connection.

Taxonomy (parts-only, no connector/fastener details):
  GRASP    — Positioning: pick up a part
  INSERT   — Coupling: translate one part into another (direction observed visually)
  PRESS    — Coupling: push parts together until seated
  SCREW    — Coupling: rotate part to engage threads or cam (screw, cam lock pin, bolt)
  REORIENT — Positioning: rotate part before connecting (curved arrow in manual image)

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


VALID_PRIMITIVES = {"GRASP", "INSERT", "PRESS", "SCREW", "REORIENT"}
CONNECTION_PRIMITIVES = {"INSERT", "PRESS", "SCREW"}   # these form BT connection goals


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
# Connection-driven action generation (replaces VLM-generated action sequences)
# ---------------------------------------------------------------------------

def _generate_actions_from_connections(connections: list[dict]) -> list[dict]:
    """Deterministically build a robot action sequence from VLM-identified connections.

    Rule: GRASP(active) → [REORIENT(active)?] → INSERT/PRESS(active, passive)
    INSERT/PRESS release the gripper, so if the same active_part appears in
    multiple connections it is re-grasped before each subsequent connection.

    No connector/fastener details (fit_type removed). Keys match BT XML attributes.
    """
    held: int | None = None
    ever_reoriented: set[int] = set()
    actions: list[dict] = []

    for conn in connections:
        active = conn.get("active_part")
        passive = conn.get("passive_part")
        prim = conn.get("primitive")

        if active is None or passive is None or prim not in CONNECTION_PRIMITIVES:
            continue

        # GRASP if not currently held
        if held != active:
            actions.append({"primitive": "GRASP", "part_id": active})
            held = active

        # REORIENT once per active_part per step
        if conn.get("needs_reorient") and active not in ever_reoriented:
            actions.append({
                "primitive": "REORIENT",
                "part_id": active,
                "description": conn.get("reorient_description") or "reorient before connecting",
            })
            ever_reoriented.add(active)

        # Emit connection primitive — keys match BT XML Action attributes
        if prim == "INSERT":
            actions.append({
                "primitive": "INSERT",
                "active_part": active,
                "passive_part": passive,
                "direction": conn.get("insertion_dir", "horizontal"),
            })
        elif prim == "PRESS":
            actions.append({
                "primitive": "PRESS",
                "active_part": active,
                "passive_part": passive,
                "direction": conn.get("insertion_dir", "downward"),
            })
        elif prim == "SCREW":
            actions.append({
                "primitive": "SCREW",
                "active_part": active,
                "passive_part": passive,
                "direction": conn.get("insertion_dir", "inward"),
            })

        held = None  # INSERT/PRESS/SCREW release the gripper

    return actions


def _connections_to_connections_formed(connections: list[dict], step: dict) -> list[dict]:
    """Convert VLM connection list to connections_formed format for BT goal tracking.

    Each entry carries:
      predicate, part1, part2      — connection identity
      active_part, passive_part    — BT action parameters
      direction                    — motion direction (visual, from insertion_dir)
      needs_reorient               — whether a reorient action precedes this connection
      reorient_description         — natural language reorient instruction
      visual_description           — NL cue for VLM-evaluated IsConnected condition node;
                                     a robot VLM reads this + camera image at runtime to
                                     verify the connection is complete (arXiv 2501.03968)
    """
    pred_map = {"INSERT": "inserted", "PRESS": "pressed", "SCREW": "screwed"}
    stage2_ctx = step.get("stage2_description", "").strip()
    result = []
    for conn in connections:
        prim = conn.get("primitive")
        if prim in pred_map:
            p1 = conn["active_part"]
            p2 = conn["passive_part"]
            direction = conn.get("insertion_dir", "downward")
            predicate = pred_map[prim]
            visual_description = (
                f"Part {p1} is {predicate} into part {p2} "
                f"(direction: {direction}). Context: {stage2_ctx}"
            )
            result.append({
                "predicate": predicate,
                "part1": p1,
                "part2": p2,
                "active_part": p1,
                "passive_part": p2,
                "direction": direction,
                "needs_reorient": conn.get("needs_reorient", False),
                "reorient_description": conn.get("reorient_description") or "reorient before connecting",
                "visual_description": visual_description,
            })
    return result


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
# Iterative refinement loop (arXiv 2409.09435)
# ---------------------------------------------------------------------------

def _refinement_loop(
    llm,
    b64_img: str,
    initial_result: dict,
    new_parts: list[int],
    max_rounds: int = 2,
) -> dict:
    """Re-prompt VLM for any new_parts absent from the initial connection list.

    If the first VLM call misses connections for some new_parts, a targeted
    follow-up is sent naming exactly which parts need connections.  New
    connections are merged into the result (no duplicates by active+passive key).

    Literature: arXiv 2409.09435 — "Behavior Tree Generation using Large
    Language Models for Sequential Manipulation Planning with Human Instructions
    and Feedback" — uses iterative feedback to improve coverage completeness.
    """
    from llm.utils import invoke_multimodal

    result = dict(initial_result)
    if result.get("parse_error"):
        return result

    for _ in range(max_rounds):
        covered = {c["active_part"] for c in result.get("connections", [])
                   if isinstance(c.get("active_part"), int)}
        missing = [p for p in new_parts if p not in covered]
        if not missing:
            break  # all parts have at least one connection

        refinement_prompt = (
            f"In your previous response you identified connections for parts "
            f"{sorted(covered)}. However, the following parts from new_parts "
            f"were NOT given any connections: {missing}.\n\n"
            f"Looking at the assembly image again, what does each of these "
            f"parts connect to, and what is the connection type "
            f"(INSERT, PRESS, or SCREW)? "
            f"Reply in the same JSON format as before, listing ONLY the missing "
            f"connections (do not repeat already-found ones)."
        )

        raw = invoke_multimodal(llm, refinement_prompt, [b64_img], mime_type="image/jpeg")
        cleaned = _clean_json(raw)
        try:
            extra = json.loads(cleaned)
        except json.JSONDecodeError:
            break  # unparseable; stop refining

        new_conns = extra.get("connections", []) if isinstance(extra, dict) else []
        if not new_conns:
            break

        existing_keys = {
            (c["active_part"], c["passive_part"])
            for c in result.get("connections", [])
            if isinstance(c.get("active_part"), int) and isinstance(c.get("passive_part"), int)
        }
        for conn in new_conns:
            ap = conn.get("active_part")
            pp = conn.get("passive_part")
            if isinstance(ap, int) and isinstance(pp, int):
                key = (ap, pp)
                if key not in existing_keys:
                    result.setdefault("connections", []).append(conn)
                    existing_keys.add(key)

    return result


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

        # --- Iterative refinement: re-prompt for any new_parts without connections ---
        # Literature: arXiv 2409.09435 (targeted follow-up for coverage completeness)
        step_actions = _refinement_loop(llm, b64_img, step_actions, new_parts)

        # --- Discard unreliable VLM-generated bookkeeping fields ---
        for key in ("step_id", "parts_involved", "preconditions", "effects"):
            step_actions.pop(key, None)

        # --- Inject authoritative tree-derived fields (before connections_formed so
        #     stage2_description is available for visual_description generation) ---
        step_actions["step_idx"] = step_idx
        step_actions["new_parts"] = new_parts
        step_actions["subassembly_parts"] = subassembly_parts
        step_actions["all_parts"] = step["all_parts"]
        step_actions["stage2_description"] = step_text

        # --- Generate action sequence and connections_formed from VLM-identified connections ---
        connections = step_actions.get("connections", [])
        step_actions["actions"] = _generate_actions_from_connections(connections)
        step_actions["connections_formed"] = _connections_to_connections_formed(connections, step_actions)

        # --- Post-VLM connection completeness check ---
        connections_formed = step_actions["connections_formed"]
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
