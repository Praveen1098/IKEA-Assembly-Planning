# Pipeline Technical Updates — Week of 2026-04-02

> **Context:** Manual2Skill (RSS 2025) is a VLM-based pipeline that parses IKEA assembly manuals and generates hierarchical assembly trees and robot-executable Behavior Trees. This document summarises the technical updates made this week across Stage 3 (action extraction), Stage 4 (BT synthesis), and the new Stage 3.5 (formal validation) and Streamlit visualisation app.

---

## 1. Stage 3 — Action Extraction (`stage3_action_extraction.py`)

### 1.1 New Primitive: SCREW

A fifth robot primitive was added to the taxonomy to handle rotational fastening (cam locks, bolts, wood screws) which are ubiquitous in IKEA furniture but were incorrectly mapped to INSERT or PRESS before.

| Primitive | Type | Trigger |
|-----------|------|---------|
| GRASP | Positioning | Pick up a part |
| INSERT | Coupling | Translate part into another |
| PRESS | Coupling | Push parts together until seated |
| **SCREW** | **Coupling** | **Rotate part to engage threads or cam mechanism** |
| REORIENT | Positioning | Rotate part before connecting |

The classification is **parts-only** — connector and fastener details (dowel IDs, cam IDs) are deliberately omitted to keep the schema stable across furniture types.

**Research basis:** Taxonomy design follows the *VLM-as-formalizer* principle — the VLM identifies connections and the code composes the action sequence, separating perception from planning.
> "Vision Language Models Cannot Plan, but Can They Formalize?" (arXiv 2509.21576)

---

### 1.2 Iterative Refinement Loop

After an initial VLM call, a **completeness check** verifies that every new part introduced in a step appears in at least one extracted connection. If any part is missing:

1. A targeted follow-up prompt is constructed, explicitly naming the missing parts.
2. The VLM is re-called (max 2 refinement rounds).
3. Results are merged — duplicates resolved by `(active_part, passive_part)` key.

This directly addresses the documented failure mode of VLMs producing incomplete object-relation graphs.

**Research basis:**
> arXiv 2409.09435 — "BT Generation using LLMs with Human Instructions and Feedback" (iterative feedback loop)
> arXiv 2509.21576 — "VLMs fail to capture exhaustive object relations"

---

### 1.3 VLM-Evaluated Condition Nodes

Each connection now carries a `visual_description` field — a natural-language sentence describing what a correct assembly state looks like from a camera view. This field is propagated into BT `Condition` nodes (as a `description` XML attribute), enabling a robot VLM to evaluate assembly state at runtime.

**Research basis:**
> arXiv 2501.03968 — "VLM-driven Behavior Trees" (Microsoft) — camera-verified condition nodes

---

### 1.4 Deterministic Action Generation from Connections

The action sequence (GRASP → REORIENT? → INSERT/PRESS/SCREW) is now generated **deterministically from the connection list**, removing the previous reliance on the VLM to output a valid action sequence. Rules:

- If a part is not currently held, emit GRASP first.
- If `needs_reorient=True` and part has not yet been reoriented in this step, emit REORIENT.
- INSERT/PRESS/SCREW release the gripper (`held = None`) — the next connection re-grasps.

This approach is consistent with the **VLM-as-formalizer** design pattern.

---

### 1.5 3-Step Chain-of-Thought Prompting

The action extraction prompt (`prompts/action_extraction.txt`) uses a 3-step CoT structure:
1. **Visual cues** — describe observed assembly motions from the manual image.
2. **Connection mapping** — map observed motions to GRASP/INSERT/PRESS/SCREW/REORIENT.
3. **Parameter extraction** — extract `active_part`, `passive_part`, `direction`, `visual_description`.

**Research basis:**
> SPAR (arXiv 2509.13691), NL2Plan (arXiv 2405.04215) — chain-of-thought decomposition for robot task planning

---

## 2. Stage 3.5 — PDDL Consistency Validation (`stage3_5_pddl_validate.py`) *(New)*

A new optional stage validates the logical consistency of `actions.json` before the BT is compiled.

### Algorithm

```
actions.json
     │
     ▼
Auto-generate PDDL domain   ← static SKILL_LIBRARY (pre/postconditions per primitive)
Auto-generate PDDL problem  ← init: all parts accessible + gripper empty
                               goal: all connections achieved
     │
     ▼
PyPerPlan (greedy best-first + hFF heuristic)
     │
     ▼
VALID  → plan found (N actions)
INVALID → diagnose which goal predicates are unreachable
```

### PDDL Skill Library

| Primitive | Preconditions | Postconditions |
|-----------|--------------|----------------|
| GRASP | accessible(?p), gripper-empty | holding(?p), ¬gripper-empty |
| INSERT | holding(?a), accessible(?p) | inserted(?a,?p), ¬holding(?a), gripper-empty |
| PRESS | holding(?a), accessible(?p) | pressed(?a,?p), ¬holding(?a), gripper-empty |
| SCREW | holding(?a), accessible(?p) | screwed(?a,?p), ¬holding(?a), gripper-empty |
| REORIENT | holding(?p) | holding(?p) |

PDDL predicates used: `holding`, `gripper-empty`, `accessible`, `inserted`, `pressed`, `screwed`.

### CLI Usage

```bash
python inference/stage3_5_pddl_validate.py --actions outputs/<timestamp>/<cat>/<name>/actions.json
```

Or integrated into the pipeline:

```bash
python inference/run.py --output_format bt --validate_pddl
```

**Research basis:**
> PMC11504948 — "Autonomous Robot Task Execution in Flexible Manufacturing: Integrating PDDL and Behavior Trees in ARIAC 2023"
> SoarGroup PDDL domain templates — `assembly.pddl` + `gripper.pddl`
> PyPerPlan (aibasel/pyperplan) — pure-Python PDDL planner

---

## 3. Stage 4 — Behavior Tree Synthesis (`stage4_formalize.py`)

Stage 4 was substantially refactored. All `py_trees`, `networkx`, and LLM dependencies were removed. The stage is now **pure Python stdlib** and runs without any external packages.

### 3.1 Goal-Oriented Backchaining BT Structure

Each assembly connection goal is encoded as a **Fallback subtree**:

```
Fallback [achieve_<predicate>_<part1>_<part2>]
├── Condition: IsConnected(part1, part2, description=<visual cue>)
└── Sequence [do_<predicate>_<part1>_<part2>]
    ├── Fallback [achieve_held_<active_part>]
    │   ├── Condition: IsHeld(part, description=<visual cue>)
    │   └── Action: Grasp(part)
    ├── Action: Reorient(part, description)   [only if needs_reorient=True]
    └── Action: Insert / Press / Screw (active_part, passive_part, direction)
```

This pattern implements the **goal-oriented backchaining** synthesis algorithm: each node's goal is its postcondition, its preconditions are checked, and a Fallback guarantees the precondition is satisfied before execution.

**Research basis:**
> M. Colledanchise & P. Ögren, "Behavior Trees in Robotics and AI", Cambridge University Press, 2018 (Chapter 7 — backchaining synthesis)
> M. Colledanchise & P. Ögren, "How Behavior Trees Modularize Hybrid Control Systems and Generalize Sequential Behavior Compositions", IEEE T-RO, 2017

---

### 3.2 Causal Graph → Parallel BT Nodes

Two connections are **causally dependent** if:
- The `active_part` of one equals the `passive_part` of another (sequencing constraint), or
- Both share the same `active_part` (must be grasped one at a time).

**Algorithm:**

1. Build a directed causal graph `G`: edge `i → j` if connection `j` depends on `i`.
2. Run **Union-Find** to partition connections into causally independent groups.
3. Connections in the **same group** → `Sequence` (order preserved by topological sort).
4. **Different groups** → wrapped in a `<Parallel success_count=N failure_count=1>` node.

This replaces the previous flat-Sequence approach with structure that exposes true assembly parallelism.

**Research basis:**
> F. Martín et al., "Flexible BT-Based Task Planning for Service Robots", AAMAS 2021 (arXiv 2101.01964)
> PlanSys2 open-source implementation: `compute_bt.cpp` (github.com/PlanSys2/ros2_planning_system)

---

### 3.3 BehaviorTree.CPP v4 XML Format

Output is a standards-compliant BT.CPP v4 XML file (`behavior_tree.xml`) with a `TreeNodesModel` section declaring all port types — compatible with **Groot2** for visual editing.

```xml
<root BTCPP_format="4">
  <BehaviorTree ID="AssembleFurniture">
    <Sequence name="assembly_root">
      <Parallel name="parallel_assembly" success_count="2" failure_count="1">
        <Fallback name="achieve_inserted_0_3"> ... </Fallback>
        <Fallback name="achieve_screwed_2_3"> ... </Fallback>
      </Parallel>
    </Sequence>
  </BehaviorTree>
  <TreeNodesModel> ... </TreeNodesModel>
</root>
```

**Research basis:**
> BTGenBot (arXiv 2403.12761, IROS 2024) — BT.CPP v4 XML format conventions and TreeNodesModel schema

---

### 3.4 `to_pddl()` Function Added

`stage4_formalize.to_pddl(actions_data, output_path)` writes `domain.pddl` and `problem.pddl` for each furniture item, delegating generation to `stage3_5_pddl_validate`. Called from the Streamlit app and `stage5_bt_compile.py`.

---

## 4. Streamlit Visualisation App (`app.py` + `app_viz.py`) *(New)*

A 9-tab Streamlit app (`inference/app.py`) provides interactive visualisations of every pipeline output. A new helper module `inference/app_viz.py` contains all visualisation logic with no LLM calls.

### Tab Structure

| Tab | Content |
|-----|---------|
| Inputs | Manual images + scene image viewer |
| Stage 1 | Part table extracted by VLM |
| Stage 2 | Assembly plan text |
| Assembly Tree | **Interactive Plotly** assembly graph (pan/zoom/hover) |
| Stage 3 — Actions | actions.json viewer |
| PDDL | Domain + problem code + Skill Library table + **Plotly action dependency graph** |
| PDDL Validation | **Run validator button** → plan timeline bar chart |
| Behavior Tree | **vis.js interactive tree** (color-coded nodes, hover tooltips) + ASCII + XML |
| BT Verification | LTL properties table + **generated nuXmv SMV model** |

---

### 4.1 Interactive Assembly Tree — Plotly (`make_plotly_assembly_tree`)

Replaces the static `matplotlib` plot. Uses `networkx` for layout positions and `plotly.graph_objects.Scatter` for:
- Edge trace (`mode='lines'`, grey)
- Node trace (`mode='markers+text'`): green for leaf parts, blue for assembly steps
- Hover shows node name and type; supports pan, zoom, click

---

### 4.2 Interactive Behavior Tree — vis.js (`bt_xml_to_visjs_html`)

Renders the BT.CPP v4 XML as a self-contained HTML page using **vis.js 9.1.9** loaded from CDN — no Graphviz binary required. Embedded via `streamlit.components.v1.html`.

Node color scheme:

| BT Node Type | Color | Shape |
|---|---|---|
| Sequence | #1565C0 (blue) | box |
| Fallback | #B71C1C (red) | diamond |
| Parallel | #6A1B9A (purple) | hexagon |
| Condition | #2E7D32 (green) | ellipse |
| Action | #E65100 (orange) | box |

Hover tooltips show all XML attributes (`description`, `direction`, `part`, etc.). Layout: hierarchical top-down, physics disabled.

---

### 4.3 PDDL Plan Timeline — Plotly (`make_plotly_plan_sequence`)

Shows the PyPerPlan output plan as a horizontal bar chart. Each bar corresponds to one PDDL action, colored by type (grasp=green, insert/press/screw=blue, reorient=yellow).

---

### 4.4 PDDL Skill Dependency Graph — Plotly (`make_pddl_skill_graph`)

Bipartite graph showing predicate → action → predicate edges (preconditions and effects) for all five primitives. Renders without running the pipeline — useful for understanding the PDDL domain at a glance.

---

### 4.5 nuXmv SMV Model Generator (`_bt_xml_to_smv_direct`)

Converts BT.CPP v4 XML directly to a nuXmv `.smv` model with **no external dependencies** — pure Python stdlib. Implements the formal BT semantics from Colledanchise & Ögren (2018), Chapter 2:

- **Condition nodes** → boolean environment variable (`TRUE : s; FALSE : f`)
- **Action nodes** → ternary nondeterministic variable (`{s, r, f}`)
- **Sequence** → `DEFINE case` expression: first failure propagates, first running blocks, last child success = success
- **Fallback** → `DEFINE case` expression: first success propagates, first running blocks, all failure = failure
- **Parallel** → conservative Sequence encoding (M-of-N simplified)

Generated `LTLSPEC` properties:
```smv
LTLSPEC G(F(root_ret = s));                          -- Goal reachability
LTLSPEC G(root_ret != f -> F(root_ret = s));         -- No permanent failure
LTLSPEC G(!(action_a = r & action_b = r));           -- Mutual exclusion
```

When `behaverify` is installed, the CLI path is tried first (`behaverify nuxmv ...`); the direct encoder is the fallback.

**Research basis:**
> D. Sprague et al., "BehaVerify: Verifying Temporal Logic Specifications for Behavior Trees", SEFM 2022 (verivital/behaverify)
> M. Colledanchise & P. Ögren, "Behavior Trees in Robotics and AI", CUP 2018 — Chapter 2 formal semantics

---

### 4.6 Predefined LTL Assembly Safety Properties

The BT Verification tab displays four domain-specific LTL properties:

| Property | Formula |
|----------|---------|
| Grasp before connect | `G((insert(A,B) \| press(A,B) \| screw(A,B)) -> P(holding(A)))` |
| Goal reachability | `G(F(all_connected))` |
| Mutual exclusion | `G(!(holding(A) & holding(B) & A != B))` |
| Gripper release after connect | `G((insert(A,B) \| press(A,B) \| screw(A,B)) -> X(gripper_empty))` |

---

## 5. Dependency and Infrastructure Changes

### `pyproject.toml`

| Package | Change | Reason |
|---------|--------|--------|
| `plotly>=5.0.0` | **Added** | Interactive charts in Streamlit app |
| `behaverify>=1.0.0` | Already declared | LTL verification via nuXmv |
| `pyperplan>=2.1` | Already declared | PDDL plan validation |

### New Files

| File | Purpose |
|------|---------|
| `inference/stage3_5_pddl_validate.py` | Stage 3.5 PDDL auto-generator + PyPerPlan validator |
| `inference/app_viz.py` | Streamlit visualisation helpers |
| `inference/app.py` | 9-tab Streamlit app |
| `inference/stage5_bt_compile.py` | Stage 5 BT compiler (pre-existing, updated) |

### `run.py` — New CLI Flags

```bash
python inference/run.py \
  --output_format bt        # tree | actions | bt
  --validate_pddl           # run Stage 3.5 after action extraction
  --model gpt4o             # model recipe ID
```

---

## 6. Reference List (for Presentation)

### Core Behavior Tree Theory

1. **M. Colledanchise & P. Ögren**, "Behavior Trees in Robotics and AI: An Introduction", Cambridge University Press, 2018.
   - Chapters 2 (formal semantics), 7 (backchaining synthesis)
   - [arxiv.org/abs/1709.00084](https://arxiv.org/abs/1709.00084)

2. **M. Colledanchise & P. Ögren**, "How Behavior Trees Modularize Hybrid Control Systems and Generalize Sequential Behavior Compositions, the Advantages and Limitations", IEEE T-RO, 2017.

### BT Synthesis from Planning

3. **F. Martín, F. Rico, J. González-Múñoz, A. Matea**, "Flexible Behavior Trees: In search of the mythical PDDL-BT correspondence", AAMAS 2021.
   - [arxiv.org/abs/2101.01964](https://arxiv.org/abs/2101.01964)
   - **Algorithm used:** causal graph → Union-Find partition → Parallel BT nodes
   - **Open-source reference:** PlanSys2 `compute_bt.cpp` (github.com/PlanSys2/ros2_planning_system)

4. **M. Iovino et al.**, "A Survey of Behavior Trees in Robotics and AI", Robotics and Autonomous Systems, 2022.
   - [arxiv.org/abs/2005.05842](https://arxiv.org/abs/2005.05842)

### VLM + LLM for Task Planning and BT Generation

5. **"Vision Language Models Cannot Plan, but Can They Formalize?"**, 2025.
   - [arxiv.org/abs/2509.21576](https://arxiv.org/abs/2509.21576)
   - **Key insight used:** VLM-as-formalizer pattern; code composes BT, VLM identifies connections

6. **"BT Generation using LLMs with Human Instructions and Feedback"**, 2024.
   - [arxiv.org/abs/2409.09435](https://arxiv.org/abs/2409.09435)
   - **Algorithm used:** iterative refinement loop for missing connections

7. **"VLM-Driven Behavior Trees for Robotic Task Planning"** (Microsoft), 2025.
   - [arxiv.org/abs/2501.03968](https://arxiv.org/abs/2501.03968)
   - **Feature used:** `description` attribute on Condition nodes for camera-based verification

8. **BTGenBot (IROS 2024)** — "Generating Behavior Trees for Robots using LLMs".
   - [arxiv.org/abs/2403.12761](https://arxiv.org/abs/2403.12761)
   - **Used for:** BT.CPP v4 XML format conventions and `TreeNodesModel` port schema

9. **SPAR** — Structured Prompting for Action-Reasoning, 2025.
   - [arxiv.org/abs/2509.13691](https://arxiv.org/abs/2509.13691)

10. **NL2Plan** — Natural Language to PDDL planning pipeline.
    - [arxiv.org/abs/2405.04215](https://arxiv.org/abs/2405.04215)

### PDDL + Formal Validation

11. **"Autonomous Robot Task Execution in Flexible Manufacturing: Integrating PDDL and Behavior Trees in ARIAC 2023"**, PMC11504948, 2024.
    - [pmc.ncbi.nlm.nih.gov/articles/PMC11504948](https://pmc.ncbi.nlm.nih.gov/articles/PMC11504948)
    - **Algorithm used:** PDDL domain auto-generation + planner for BT consistency check

12. **SoarGroup PDDL Domain Templates** — `assembly.pddl`, `gripper.pddl`.
    - [github.com/SoarGroup/Domains-Planning-Domain-Definition-Language](https://github.com/SoarGroup/Domains-Planning-Domain-Definition-Language)

13. **PyPerPlan** — pure-Python PDDL planner.
    - [github.com/aibasel/pyperplan](https://github.com/aibasel/pyperplan)

### LTL Verification of Behavior Trees

14. **D. Sprague, L. Feng, M. Lahijanian**, "BehaVerify: Verifying Temporal Logic Specifications for Behavior Trees", SEFM 2022.
    - [verivital/behaverify](https://github.com/verivital/behaverify)
    - **Tool used:** `behaverify nuxmv` CLI for BT → nuXmv SMV model generation

### Tooling

15. **BehaviorTree.CPP v4** — C++ BT library used as output format.
    - [github.com/BehaviorTree/BehaviorTree.CPP](https://github.com/BehaviorTree/BehaviorTree.CPP)
    - **Groot2** — visual editor compatible with generated XML

16. **Manual2Skill (RSS 2025)** — this project.
    - IKEA assembly planning from manuals using VLMs

---

## 7. Architecture Diagram (Pipeline Flow)

```
IKEA Manual PDF
      │
      ▼
Stage 1: Part Identification (VLM)
      │  → part table: {id: name}
      ▼
Stage 2: Step-by-Step Planning (VLM)
      │  → assembly plan text
      ▼
Convert: Tree JSON Builder (VLM)
      │  → tree.json (nested assembly dependency tree)
      ▼
Stage 3: Action Extraction (VLM + deterministic)      ← NEW: SCREW, refinement loop, CoT
      │  → actions.json {furniture, parts, all_connections, steps}
      ▼
Stage 3.5: PDDL Validation (optional)                 ← NEW: auto PDDL + PyPerPlan
      │  → VALID / INVALID + diagnosis
      ▼
Stage 4: BT Synthesis (deterministic, stdlib only)    ← NEW: causal graph, Parallel nodes
      │  → behavior_tree.xml (BT.CPP v4, Groot2-compatible)
      │  → behavior_tree.txt (ASCII preview)
      │  → domain.pddl + problem.pddl
      ▼
Stage 5: BT Compilation (BehaviorTree.CPP)            ← pre-existing
      │  → compiled BT
      ▼
Streamlit App (9 tabs)                                ← NEW: Plotly, vis.js, nuXmv
      → Interactive visualisations of all outputs
```

---

*Generated: 2026-04-02*
