"""reward_bt_synth — end-to-end reward-grounded BT synthesis for IKEA assembly.

Two contributions:
  1. AST-level STRIPS action-model extraction from robosuite-convention reward
     code (ast_extractor.py).
  2. Coverage-Conservation Structured Plan Extraction replacing Stage 2
     (stage2_v2_plan.py).

No PDDL, no pyperplan, no LLM-guessed action models, no repair heuristics in
the BT-synthesis path. See C:\\Users\\pmsekar\\.claude\\plans\\wild-juggling-piglet.md
for the full design rationale.
"""
