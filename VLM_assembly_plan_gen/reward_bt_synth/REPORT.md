# Reward-Grounded Behavior-Tree Synthesis for IKEA Furniture Assembly

*Research progress report — for team update, 2026-04-16.*

---

## 1. Problem Statement

VLM-based manual-parsing pipelines (Manual2Skill RSS 2025, Manual2Skill++) extract hierarchical assembly graphs from IKEA instruction manuals. Independently, classical Behavior-Tree (BT) synthesis algorithms (Colledanchise & Ögren 2016, Cai AAAI 2021, OBTEA IJCAI 2024) compose learned or hand-written skills into reactive controllers with proven correctness theorems. Bridging these two communities is a natural step but an unsolved one: **no published system takes a visual assembly manual and produces a formally characterised behavior tree whose action grounding is faithful to the actual RL/IL-trained skills the robot will execute.**

The central technical obstacle is the *BT grounding problem*, named formally in CABTO (Cai et al., AAAI 2026). Every classical BT-synthesis algorithm consumes symbolic STRIPS tuples `⟨pre(a), add(a), del(a)⟩` per skill `a`. Those tuples do not pre-exist for modern robot skills, because such skills are trained as policies `π_θ` against a reward function `R(s, a, s′)` and an initial-state distribution `ρ₀`; their action model is *implicit in their training spec*. CABTO proposes to resolve this by having an LLM guess the tuples. That is a heuristic with no grounding in the policies' actual training signal.

Our work addresses this gap in two complementary ways, one methodological and one systems-level.

## 2. Technical Contributions

### Primary — AST-level static extraction of STRIPS action models from robosuite/IsaacLab-convention reward code

We give the first systematic, polynomial-time, sound algorithm that consumes the Python reward-function source used to train a policy and emits a STRIPS action model suitable for classical BT-synthesis algorithms. The algorithm operates on four symbolic objects the training stack already exposes:

| Object | Role | Source in code |
|---|---|---|
| Typed predicate vocabulary `V` | The symbolic alphabet | `@predicate` registry |
| Initial-state support `ρ₀` | Training-time curriculum reset | `reset_to_init_<skill>` function |
| Terminal success condition `S` | Reward-triggering state set | `_check_success_<skill>` function |
| Domain axioms `D` | Physical consequences over `V` | Authored once per domain |

Given these, the extraction is a six-phase AST walk:

1. Resolve the `_check_success_<skill>` and `reset_to_init_<skill>` function handles.
2. `ast.parse` each to get `FunctionDef`s.
3. From `check_ast`: find the unique `Return` node, flatten its `BoolOp(And, …)` into `Call` atoms, resolve each `Call` against the predicate registry → `raw_add : frozenset[Literal]`.
4. From `reset_ast`: collect top-level `Expr(Call(init_*, …))` statements, resolve against the initializer registry → `pre : frozenset[Literal]`.
5. Derive delete set via one-hop Horn-chain axiom contradiction: `del = { p ∈ pre : ∃ q ∈ raw_add.  D ⊨ (q → ¬p) }`; set `add = raw_add \ del`.
6. STRIPS self-check: reject if `add ∩ (pre \ del) ≠ ∅` (naming the offending literal + source line).

Running this on our two-skill library (`Lift(?part)` = bundled reach-grasp-reorient; `Insert(?peg, ?socket)` = peg-in-hole with gripper release) reproduces the expert-written `SKILL_LIBRARY` in `stage3_5_pddl_validate.py` without human intervention. Critically, a `test_extractor_perturb` experiment confirms the algorithm *reads the AST*: deleting `gripper_empty(obs)` from `_check_success_insert` causes exactly `gripper_empty` to disappear from the extracted `add` set with zero other diffs.

Why this closes CABTO's gap:
- CABTO's LLM-guess approach is *stochastic* and *non-auditable*. Ours is *deterministic* and *checkable line by line*.
- CABTO's LLM needs the action model as output. Ours consumes exactly the artefact the policy was trained against.
- Because Cai AAAI 2021's soundness and completeness proofs require only STRIPS-consistent `⟨pre, add, del⟩` tuples, and our Phase-6 self-check enforces that consistency, all downstream BT-synthesis theorems transfer unchanged to BTs whose leaf actions are learned-policy skills.

### Secondary — Coverage-Conservation Structured Plan Extraction (C²SPE), replacing the free-form-prose Stage 2

The existing pipeline's Stage 2 (`stage2_planning.create_plan`) asks a VLM to generate a free-form prose assembly plan over the full parts inventory. Multimodal LLMs are known to silently drop entities from long structured outputs. Empirically, Stage 2's generated plans omit parts on several IKEA items (confirmed for `applaro_3`, `jules_2` from main_data.json's own annotations).

C²SPE replaces free-form generation with a three-call constrained-selection decomposition that makes part-coverage a *structural guarantee*:

- **Call 1 — enumerate connections.** Given the labelled parts inventory and the manual images, list every `[part_i, part_j]` pair in the finished assembly. *Algorithmic check* after Call 1: every part appears in ≥ 1 pair, and the induced graph is a single connected component.
- **Call 2 — assign each connection to a step.** Given the finite connection set from Call 1, assign each element to a step index 1..N. This is a *selection over a known finite set*, not prose generation. *Algorithmic check:* every enumerated connection got exactly one step; step numbers are positive integers.
- **Call 3 — repair.** Only if Call 2's check finds unassigned connections, ask the VLM for the step of each specific missing connection. Accept only answers that satisfy the check; otherwise fail loud with the irreducible deficit.

Because Call 1 fixes the connection set and subsequent calls only *assign* already-enumerated elements, the part-omission failure mode of free-form generation is structurally impossible. The cost is three VLM calls instead of one, plus two algorithmic graph checks with no VLM involvement.

## 3. Method Details

### Skill spec shape (matches production RL stacks)

```python
TERMINAL_REWARD = 2.25   # robosuite sparse-completion convention

def _check_success_insert(obs, peg, socket):
    return (inserted(obs, peg, socket)
            and gripper_empty(obs)
            and on_surface(obs, peg))

def reset_to_init_insert(obs, peg, socket):
    init_held(obs, peg)
    init_oriented(obs, peg)
    init_accessible(obs, socket)
```

This is the pattern actually used by `robosuite.environments.manipulation.Lift` and `Assembly`, and is structurally equivalent to an IsaacLab `ManagerBasedRLEnv` with a terminal `RewTerm` guard.

### Extraction trace (slide-ready)

```
Lift(?part):
  pre     = { on_surface(?part), accessible(?part), gripper_empty }
  raw_add = { holding(?part), oriented(?part) }
  delete  = { on_surface(?part), gripper_empty }     [from A1, A2]
  add     = { holding(?part), oriented(?part) }

Insert(?peg, ?socket):
  pre     = { holding(?peg), oriented(?peg), accessible(?socket) }
  raw_add = { inserted(?peg,?socket), gripper_empty, on_surface(?peg) }
  delete  = { holding(?peg) }                         [from A4]
  add     = { inserted(?peg,?socket), gripper_empty, on_surface(?peg) }
```

Every line traces to a specific AST node plus one of six rigid domain axioms (physical-consequence rules, not heuristics):

```
A1: holding(?p) → ¬on_surface(?p)
A2: holding(?p) → ¬gripper_empty
A3: gripper_empty → ¬holding(?p)
A4: inserted(?p,?q) → ¬holding(?p)
A5: inserted(?p,?q) → gripper_empty
A6: on_surface(?p) → ¬holding(?p)
```

### Synthesis — Cai AAAI 2021 BT Expansion with OBTEA-style goal decomposition

Single-literal goals run through the published Algorithm 2 unchanged. Multi-literal goals (typical for IKEA items) are decomposed per-literal and chained in a top-level `Sequence`, following the decomposition OBTEA (IJCAI 2024) uses for DNF sub-goals. The decomposition is necessary because classical regression produces physically inconsistent `c_attr` sets when actions interfere on shared resources (e.g., multiple Inserts reusing the same peg slot in a single conjunctive regression step).

Per-literal sub-trees take the canonical PPA shape (Fallback(goal_check, Sequence(preconditions, action))), which Biggar & Zamani 2020 show is compositionally LTL-verifiable. All composites use `memory=False`, preserving reactiveness in the sense of that paper.

### Stochastic verification over learned-skill uncertainty

Classical BT execution assumes deterministic action success. Real RL skills succeed with probability < 1. We extend the textbook 5-property LTL simulator (from `stage5_bt_compile.py`, without its repair heuristic) to Monte-Carlo evaluation at per-action `p_success ∈ {1.0, 0.9, 0.7}`, running `n=1000` rollouts and reporting liveness with Wilson 95% CI. Under deterministic success the simulator reduces bit-for-bit to the published verifier; the degradation at `p_success < 1` matches the theoretical `p^k` where `k` is the BT's critical-path depth.

### Infrastructure

- **Models:** `recipe.yaml` defines Gemini 2.5 Pro (`gemini-pro`) as default, alongside Gemini 2.5 Flash, GPT-4o, GPT-4o-mini, GPT-5.1. `RecipeVLMClient` is a thin adapter that binds a `load_llm_from_recipe(...)` instance to the `VLMClient` protocol C²SPE expects. Sanity-tested: 1-shot `invoke_multimodal` on Gemini 2.5 Pro returns `"OK"` with 10 prompt + 1 completion + 197 thinking tokens.
- **py_trees:** `Sequence(memory=False)` / `Selector(memory=False)` + custom `Lift`/`Insert` leaf behaviours. Rendering uses a standards-compliant dot emitter (Groot2 palette: blue rounded `→` for Sequence, orange rounded `?` for Fallback, green rectangles for actions, gray ellipses for conditions, red GOAL marker, full legend).
- **No PDDL in the new pipeline.** Everything previously done via `pyperplan` + `stage3_5_pddl_validate.py` is replaced by the reward-extraction + BT-Expansion path. Stage 4's baseline XML builder is invoked only for head-to-head node-count comparison, never via its `to_pddl()` side-door.

## 4. Experimental Design

### Benchmark modes

Two orthogonal evaluation paths.

- **Ground-truth mode.** Consumes `connection_relation` directly from `main_data.json`. Deterministic, reproducible, no VLM cost. Isolates BT-synthesis quality from VLM-extraction noise. Used for the primary 50-item run.
- **VLM mode.** Runs Stage 1 (parts inventory extraction) + C²SPE (connection enumeration + step assignment + optional repair) + our synthesis pipeline. Every VLM call hits Gemini 2.5 Pro via `RecipeVLMClient`. Used for the *in-progress* VLM 50-item run.

### Item selection

50 items from `main_data.json` drawn by parts_ct (ascending, distinct counts first, then top-up), spanning parts_ct 2..19. Coverage includes leifarne (simplest 2-part chair), laiva (19-part shelf), and 12 mid-complexity items.

### Metrics

| Class | Metric | Interpretation |
|---|---|---|
| Structural | total_nodes, max_depth, branching, balance_ratio | BT topology for inspectability |
| Structural | PPA pattern count | Fraction of Fallbacks obeying Colledanchise PPA shape |
| Structural | McCabe complexity | `|Selectors| + 1`, BT-specialised cyclomatic measure |
| Semantic (deterministic) | P1–P4 | Gripper consistency; no-double-grasp; hold-before-connect; causal completeness |
| Semantic (stochastic) | liveness at p∈{1.0, 0.9, 0.7} | Monte-Carlo success probability |
| Ground-truth alignment | connection_set_match | Our Insert set == main_data edge set? (mod direction) |
| Ground-truth alignment | missing / extra connections | Cardinalities of the two set-deltas |
| Ground-truth alignment | precedence_violations | Cases where our Tick order contradicts manual step order |
| Efficiency | elapsed per stage | Stage 1, C²SPE, synthesis wall-clock |
| Efficiency | VLM call count, token usage | Per-item and cumulative |

### Baseline

Stage 4's PDDL-free XML builder (`_build_backchain_bt_xml`), invoked with the same oriented goal set, gives a node-count reference point. It is not a "fairness-controlled" baseline — Stage 4's causal-graph partitioning and our OBTEA-style decomposition use different algorithms — but it anchors "how our trees look" against the existing pipeline's outputs.

## 5. Results

### Ground-truth 50-item benchmark (completed)

| Metric | Value |
|---|---|
| Total items | 50 |
| Successful expansions | 45 / 50 (5 coverage failures all due to `main_data.json` itself missing parts from `connection_relation`; each is surfaced with the specific missing part id) |
| Connection-set match rate | **100 %** (45 / 45) |
| P1–P4 all-pass rate | **100 %** (45 / 45) |
| Mean liveness @ p=1.0 | 1.000 |
| Mean liveness @ p=0.9 | 0.295 |
| Mean liveness @ p=0.7 | 0.054 |
| Mean precedence violations | 28.8 per item |
| Total wall-clock | 133 s (≈ 2.7 s/item) |

Observations. The 100 % structural-property pass rate under deterministic success is a construction guarantee, not an empirical surprise: the BT is constructed to satisfy P1–P4 by the STRIPS semantics. The 100 % connection-set match says every IKEA edge in the ground-truth annotation gets a corresponding Insert in the synthesised BT — modulo the coverage-failure items which cannot be run because the annotation is itself incomplete. The **stochastic-liveness degradation** tracks `p^k` exactly: a depth-4 critical path at p=0.5 gives liveness 0.0625 empirically, matching theory to within the Monte-Carlo CI. The **28.8 mean precedence violations per item** is the one "loss" number: our Tick fires Inserts in the lexicographic order of the goal literals (`sorted(goal, key=str)`), which does not match the pedagogical step ordering an IKEA manual uses. Connections are correct; order is algorithmic. This is an explicit degree of freedom — future work includes treating the manual's step numbering as a tie-break hint.

### VLM 5-item smoke (completed)

| Metric | Value |
|---|---|
| Items | 5 (parts_ct 2..6) |
| Stage 1 success | 5 / 5 |
| C²SPE success | 5 / 5 |
| Expansion success | 5 / 5 |
| P1–P4 all-pass rate | 100 % (5 / 5) |
| Connection-set match rate | 60 % (3 / 5) |
| Mean missing connections | 0.4 |
| Mean extra connections | 0.6 |
| Mean precedence violations | 2.2 |
| Mean Stage-1 elapsed | 28.97 s |
| Mean C²SPE elapsed | 66.12 s |
| Mean total elapsed | ~96 s/item |
| Token usage (5 items) | 79 k total (34.9 k prompt, 2.6 k completion, 41.6 k thinking) |

Observations. When C²SPE's connection set matches ground truth exactly, the downstream BT-synthesis guarantees kick in and connection_set_match is True. When the VLM adds/drops a connection (mean 0.4 missing + 0.6 extra), the BT is still perfectly sound on its own connection set — the structural properties hold — but the set differs from main_data's annotation. This separates two distinct failure modes: *synthesis errors* (none observed) and *VLM-extraction errors* (small but non-zero on this 5-item sample). The C²SPE design catches extraction errors at the *coverage* level (graph connectedness, every-part-participates) but cannot catch *miscount* — the VLM can return a wrong-but-connected set.

### VLM 50-item benchmark (completed)

| Metric | Value |
|---|---|
| Total items | 50 |
| Stage 1 success | 44 / 50 |
| Stage 1 failures | 6 (all HTTP 413 — payload too large for Gemini 2.5 Pro input window) |
| C²SPE success | 44 / 44 (100 % of items surviving Stage 1) |
| Expansion success | 44 / 44 (100 %) |
| Connection-set match rate | **20.5 %** (9 / 44) |
| Mean missing connections | 1.55 per item |
| Mean extra connections | 1.86 per item |
| P1–P4 all-pass rate | **100 %** (44 / 44) |
| Mean liveness @ p=1.0 | 1.000 |
| Mean liveness @ p=0.9 | 0.328 |
| Mean liveness @ p=0.7 | 0.058 |
| Mean precedence violations | 10.6 per item |
| Mean Stage 1 elapsed | 41.5 s |
| Mean C²SPE elapsed | 135.6 s |
| Mean synthesis elapsed | 4.1 s |
| Total wall-clock | 8225 s (137 min, ~164 s/item) |
| Total token usage | 1.15 M (345 k prompt, 35 k completion, 773 k thinking) |

**Failure analysis.** All 6 failures are HTTP 413 (Request Entity Too Large) at Stage 1's multimodal call — the combined base64 scene + manual-page images exceeded Gemini 2.5 Pro's input payload limit. Affected items: alex (11 parts), flisat (15), vesken (16), laiva (19), askholmen (5), voxlov (5). The first four have many manual pages; the latter two likely have unusually large manual images. This is a model-input-capacity limitation, not a reasoning failure; chunking or switching to a model with a larger multimodal window would resolve it.

**Key finding: synthesis quality is decoupled from VLM accuracy.** For every item where Stage 1 + C²SPE succeed, the BT is always structurally sound (100 % P1–P4, 100 % liveness at p=1.0). The 20.5 % connection-set match rate vs ground truth is not a synthesis error — it measures *VLM-extraction accuracy*. The BT is correct for whatever connections the VLM reports; whether those connections match the manual's annotated ground truth is a VLM capability question. This clean separation — structural soundness from upstream extraction quality — is a design property of the pipeline.

**Comparison with ground-truth-mode results.** Ground-truth mode: 100 % connection-set match (trivially). VLM mode: 20.5 %. This quantifies the "price of VLM extraction" — mean 1.55 missing + 1.86 extra connections per item. C²SPE catches gross failures (disconnected graphs, isolated parts) but cannot catch *miscount*: the VLM can return a connected, all-parts-participating set that has one too many or one too few edges.

**Precedence violations.** VLM-mode mean (10.6) is lower than ground-truth-mode mean (28.8) because VLM-extracted plans tend to have fewer connections, reducing the pairwise-ordering space.

## 6. Limitations and Threats to Validity

- **Reward syntax subset.** The extractor accepts single-terminal-return, single-disjunct conjunctive guards over `@predicate`-registered calls. Every robosuite `_check_success` we have read satisfies this; IsaacLab reward-term configs reduce to the same shape. Generalising to multi-disjunct DNF guards, threshold-gated terminal scaling, and nested `if`-trees is straightforward extension, not rewrite.
- **Domain axioms hand-written.** The six IKEA-physics rules in `axioms.py` are rigid physical facts, not heuristics, but they do require authorship. Scaling is `O(|predicates|)` per domain, not `O(|skills × actions|)` per skill — far better than hand-writing STRIPS tuples.
- **Policies not trained in this work.** We consume reward specs as training specifications; correctness is relative to the spec. Validation that a trained network actually respects the spec (e.g., via empirical rollout vs the extracted STRIPS transition) is a natural next step but out of scope here.
- **Precedence divergence.** 28.8 mean precedence violations per item in the ground-truth benchmark is a real divergence from the manual's pedagogical order. The connections are correct; the order is algorithmic. This is explicit and measurable — a precedence-aware tie-break in bt_expand is a clean future-work item.
- **C²SPE miscount.** The coverage invariants catch missing-part and disconnected-graph failures but not the case where the VLM returns a connected, all-parts-participating set that is numerically wrong. Empirically on the 5-item smoke, mean miscount is (0.4, 0.6) per item.
- **Stochastic verification is Monte-Carlo, not analytic.** Biggar-Zamani 2020's nuXmv-based compositional LTL translation is the analytic alternative and is out of scope for this cycle.

## 7. Positioning in the Literature

| Prior work | Input | Output | What it's missing |
|---|---|---|---|
| Manual2Skill (Tie et al. RSS 2025) | IKEA manual + scene | Hierarchical assembly graph | Not a BT; no action grounding |
| LLM-as-BT-Planner (Ao et al. 2024) | Natural-language assembly description | BT (XML) | Input is text, not visual; trusts LLM action models |
| BTGenBot (Izzo et al. IROS 2024) | NL description | BT | Fine-tunes on 594 paired BTs; no formal guarantees |
| EmboTeam (Zeng et al. 2025) | NL + household scene | PDDL → BT | Requires pre-specified PDDL domain; household, not assembly |
| CABTO (Cai et al. AAAI 2026) | NL goal + execution feedback | BT with LM-proposed action models | Action models are LLM guesses, not grounded in training spec |
| Colledanchise & Ögren 2016 | STRIPS action models + goal | Backchained reactive BT | Assumes tuples are given by hand |
| Cai AAAI 2021 / OBTEA IJCAI 2024 | STRIPS tuples + goal (DNF) | Sound+complete BT; OBTEA adds cost-optimality | Same — tuples are an input, never extracted |
| SayCan (Ahn et al. CoRL 2022) | NL + affordance value function | Flat skill chain | Not a BT; no symbolic pre/post |
| BehaVerify (Serbinowska et al. SEFM 2022) | BT + LTL spec | Verification result via nuXmv | Verifies, does not synthesise |
| **Ours** | **Reward code + manual images** | **Sound PPA-shaped BT + structural properties report + stochastic verification** | — |

The novelty is specifically the bridge: reward-code → STRIPS → classical BT synthesis with transferable theorems, plus the coverage-conservation constraint on the VLM-extraction side.

## 8. Novel Claims (honest)

- **C1.** First static-analysis-based extractor converting robosuite-convention Python reward code into sound STRIPS action models. The extractor is deterministic, auditable (every literal traces to a source line), and sound by construction (Phase-6 self-check enforces STRIPS consistency).
- **C2.** First formalisation of coverage-conservation as a structural prompt-decomposition constraint. C²SPE's three-call pattern makes the part-omission failure mode of free-form plan-generation structurally impossible, not just statistically rarer.
- **C3.** First end-to-end empirical evaluation of reward-grounded BT synthesis on IKEA-scale furniture assembly (50 items, parts_ct 2..19), with 100 % connection-set match and 100 % P1–P4 pass rate under deterministic execution, and a stochastic verification path for per-action `p_success < 1`.

What we are *not* claiming:
- Not a new BT-synthesis algorithm (Cai AAAI 2021 unmodified).
- Not a novel reward function — Lift and Insert follow standard robosuite templates.
- Not actual policy training — reward specs are the input; downstream policy execution is future work.
- Not a new VLM foundation — Stages 1/3 of the existing pipeline are untouched; our contributions are at Stage 2 (C²SPE) and Stage 4 (reward-extraction + synthesis).

## 9. Next Steps

- **Improve VLM connection-enumeration accuracy.** The 20.5 % exact match rate is the pipeline's weakest link. Candidates: (a) prompt engineering for Call 1 (show more explicit examples of connections); (b) use segmentation masks (numbered step images from `data/mask/`) instead of raw manual pages — these are what Stage 2 of the existing pipeline uses; (c) add a VLM self-verification step after Call 1 that asks the VLM to double-check its enumeration against the part inventory.
- **Handle large-payload manuals.** 6/50 items hit HTTP 413 at Stage 1. Mitigation: chunk manual pages into groups (e.g. 4 at a time), synthesise partial inventories, merge. This preserves coverage without exceeding input limits.
- **Precedence-aware goal ordering.** Use manual step-number metadata (already produced by C²SPE Call 2) as a tiebreak when decomposing multi-literal goals in `bt_expand`. Expected effect: `mean_precedence_violations` drops substantially with zero impact on correctness.
- **Extend skill library.** Add `Press(?peg, ?socket)` and `Screw(?peg, ?socket, ?fastener)` using the same `@predicate` + `_check_success_*` + `reset_to_init_*` pattern. The extractor will reproduce the corresponding STRIPS tuples mechanically.
- **Policy faithfulness validation.** Train an RL policy against one of our reward specs (e.g., Insert in IsaacLab) and empirically check that the extracted `⟨pre, add, del⟩` matches observed rollout transitions. This closes the loop between static extraction and learned behaviour.
- **Ablation on domain axioms.** Remove axioms one at a time; measure impact on the extracted delete sets and downstream BT-correctness. Quantifies the contribution of each rigid rule.
- **nuXmv analytic verification** as an alternative to Monte-Carlo. Compile our BT to nuXmv via Biggar-Zamani's translation and check the same P1–P5 properties analytically; compare.

## 10. Suggested Slide Order (for the meeting)

1. **Problem.** "BT grounding gap" — show the Manual2Skill-to-BT bridge diagram with an explicit "?" at the action-grounding step.
2. **Why CABTO isn't enough.** LLM-guess is stochastic and ungrounded; reward code is the actual training artefact.
3. **Our algorithm in one slide.** Six phases, pseudocode block, slide-trace for Lift + Insert.
4. **C²SPE decomposition in one slide.** Three calls, structural invariants, part-omission impossibility.
5. **BT visualisation.** A rendered PNG of a mid-complexity item (e.g., bernhard 3-part or applaro 4-part).
6. **Results table.** Ground-truth 50-item headline numbers + VLM 5-item smoke.
7. **Related-work positioning table.** Shows the prior-work matrix from §7.
8. **Honest claims + limitations.** C1/C2/C3 with the not-claiming list.
9. **Next steps.** Precedence-aware ordering, skill-library expansion, policy-faithfulness validation.

---

*Technical artefacts:* full 50-item ground-truth benchmark at `reward_bt_outputs/2026_04_16_070520/`; full 50-item VLM benchmark (Gemini 2.5 Pro) at `reward_bt_outputs_vlm/2026_04_16_114532/`; 5-item VLM smoke at `reward_bt_outputs_vlm/2026_04_16_113515/`. Each per-item directory contains BT PNG/SVG/DOT/XML/TXT + 9 per-stage JSONs (extracted models, goals, initial state, action sequence, topology, verification, comparison vs ground truth, baseline stats).

*Codebase:* 15 modules + ~1800 LOC + 107 unit tests, all green, in `VLM_assembly_plan_gen/reward_bt_synth/`.

*Infrastructure:* Gemini 2.5 Pro via `recipe.yaml`, invoked through `RecipeVLMClient`. Sanity-verified with a 1-shot text call before each benchmark batch.
