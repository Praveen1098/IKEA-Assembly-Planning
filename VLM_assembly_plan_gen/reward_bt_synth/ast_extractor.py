"""AST-level extraction of STRIPS action models from robosuite-convention reward code.

The primary contribution of this package. Given a skill name like "lift", the
extractor resolves two Python functions from reward_specs.py —

  _check_success_<skill>(obs, *formals) -> bool     # terminal guard
  reset_to_init_<skill>(obs, *formals) -> None      # initial-state sampler

— walks their ASTs, resolves each Call against the @predicate / @initializer
registries from vocabulary.py, derives the STRIPS delete set via one-hop
Horn-chain axiom contradiction, and returns an ActionModel(pre, add, delete).

Soundness is by construction (Phase 6 explicit self-check). If any phase
cannot complete, a named ExtractionError is raised carrying the source
lineno and, where applicable, the offending predicate name.

No LLM calls, no heuristics, no repair — a failure is a failure.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import ModuleType
from typing import Iterable

from .action_model import ActionModel, Literal, LiteralArg
from .axioms import Axiom, IKEA_AXIOMS, entails
from .exceptions import ExtractionError
from .vocabulary import (
    INITIALIZER_REGISTRY,
    PREDICATE_REGISTRY,
    PredicateTemplate,
)


# ---------- Public entry point ----------

def extract_action_model(
    skill_name: str,
    reward_module: ModuleType,
    axioms: Iterable[Axiom] = IKEA_AXIOMS,
) -> ActionModel:
    """Extract a STRIPS action model for `skill_name` by static analysis.

    Args:
        skill_name: suffix of `_check_success_<skill>` / `reset_to_init_<skill>`.
        reward_module: module containing the two functions.
        axioms: domain axioms used for delete-set derivation.

    Returns:
        ActionModel with frozensets for pre/add/delete and a STRIPS-consistency
        self-check passed.

    Raises:
        ExtractionError (with skill_name, phase, lineno, predicate) on any failure.
    """
    axiom_tuple = tuple(axioms)

    # Phase 1 — Resolve function handles
    check_fn = _resolve_function(
        reward_module, f"_check_success_{skill_name}", skill_name, phase=1
    )
    reset_fn = _resolve_function(
        reward_module, f"reset_to_init_{skill_name}", skill_name, phase=1
    )

    # Phase 2 — Parse ASTs
    check_def = _parse_function_def(check_fn, skill_name, phase=2)
    reset_def = _parse_function_def(reset_fn, skill_name, phase=2)

    # Formal parameters (skip the implicit first `obs` arg; reset has no `action`)
    check_formals = _skill_formal_params(check_def, has_action_arg=False)
    reset_formals = _skill_formal_params(reset_def, has_action_arg=False)
    if check_formals != reset_formals:
        raise ExtractionError(
            f"_check_success_{skill_name} formals {check_formals} != "
            f"reset_to_init_{skill_name} formals {reset_formals}",
            skill_name=skill_name,
            phase=2,
            lineno=check_def.lineno,
        )

    # Phase 3 — Extract raw_add from the terminal guard in _check_success_*
    raw_add = _extract_raw_add(check_def, check_formals, skill_name)

    # Phase 4 — Extract pre from reset_to_init_* body
    pre = _extract_pre(reset_def, reset_formals, skill_name)

    # Phase 5 — Derive delete via axiom contradiction
    delete = _derive_delete(pre, raw_add, axiom_tuple)
    add = raw_add - delete

    # Phase 6 — STRIPS self-check
    residual = (add & pre) - delete
    if residual:
        offender = sorted(str(x) for x in residual)[0]
        raise ExtractionError(
            f"STRIPS consistency violated: add ∩ (pre \\ delete) non-empty "
            f"({offender}); reward spec contradicts initial-state distribution",
            skill_name=skill_name,
            phase=6,
            predicate=offender,
        )

    return ActionModel(
        name=skill_name,
        params=check_formals,
        pre=pre,
        add=add,
        delete=delete,
    )


# ---------- Phase 1 helper ----------

def _resolve_function(
    module: ModuleType, name: str, skill_name: str, *, phase: int
):
    fn = getattr(module, name, None)
    if fn is None:
        raise ExtractionError(
            f"function {name!r} not found in module {module.__name__!r}",
            skill_name=skill_name,
            phase=phase,
        )
    if not callable(fn):
        raise ExtractionError(
            f"{name!r} in module {module.__name__!r} is not callable",
            skill_name=skill_name,
            phase=phase,
        )
    return fn


# ---------- Phase 2 helpers ----------

def _parse_function_def(fn, skill_name: str, *, phase: int) -> ast.FunctionDef:
    try:
        src = inspect.getsource(fn)
    except OSError as exc:  # pragma: no cover — only triggers for interactive defs
        raise ExtractionError(
            f"cannot read source of {fn.__name__!r}: {exc}",
            skill_name=skill_name,
            phase=phase,
        ) from exc
    src = textwrap.dedent(src)
    try:
        module_ast = ast.parse(src)
    except SyntaxError as exc:
        raise ExtractionError(
            f"AST parse failed on {fn.__name__!r}: {exc.msg}",
            skill_name=skill_name,
            phase=phase,
            lineno=exc.lineno,
        ) from exc
    if not module_ast.body or not isinstance(module_ast.body[0], ast.FunctionDef):
        raise ExtractionError(
            f"top-level of {fn.__name__!r} is not a function definition",
            skill_name=skill_name,
            phase=phase,
        )
    return module_ast.body[0]


def _skill_formal_params(
    fn_def: ast.FunctionDef, *, has_action_arg: bool
) -> tuple[str, ...]:
    """Return the skill's formal parameters with '?' prefix.

    Skips the implicit first `obs` argument. The reward_<skill> function in
    robosuite convention also takes `action` after `obs`; our check / reset
    functions do not, so has_action_arg=False is correct here.
    """
    args = fn_def.args.args
    if len(args) < 1:
        raise ExtractionError(
            f"{fn_def.name!r} must take at least `obs` as first parameter",
            skill_name=fn_def.name,
            phase=2,
            lineno=fn_def.lineno,
        )
    skip = 1 + (1 if has_action_arg else 0)
    return tuple(f"?{a.arg}" for a in args[skip:])


# ---------- Phase 3 helpers — terminal guard → raw_add ----------

def _extract_raw_add(
    fn_def: ast.FunctionDef,
    formals: tuple[str, ...],
    skill_name: str,
) -> frozenset[Literal]:
    """Resolve the unique `return <BoolOp(And, ...) | Call>` into raw_add."""
    returns = [n for n in ast.walk(fn_def) if isinstance(n, ast.Return)]
    if len(returns) == 0:
        raise ExtractionError(
            f"_check_success_{skill_name}: no return statement found",
            skill_name=skill_name,
            phase=3,
            lineno=fn_def.lineno,
        )
    if len(returns) > 1:
        raise ExtractionError(
            f"_check_success_{skill_name}: expected exactly one return, "
            f"found {len(returns)} (multi-branch terminal is out of scope)",
            skill_name=skill_name,
            phase=3,
            lineno=returns[1].lineno,
        )
    guard = returns[0].value
    if guard is None:
        raise ExtractionError(
            f"_check_success_{skill_name}: return statement has no value",
            skill_name=skill_name,
            phase=3,
            lineno=returns[0].lineno,
        )
    atoms = _flatten_and(guard, skill_name)
    return frozenset(
        _resolve_call(call, PREDICATE_REGISTRY, formals, skill_name, phase=3)
        for call in atoms
    )


def _flatten_and(node: ast.expr, skill_name: str) -> list[ast.Call]:
    """Flatten a BoolOp(And, ...) expression into a list of Call atoms.

    Accepts either a single Call (single-atom conjunction) or a recursively
    nested BoolOp(And). Anything else — Or, Compare, BinOp, UnaryOp,
    Constant, Name — raises ExtractionError. Restricts the guard to a pure
    conjunction of registered-predicate calls, which is the robosuite
    convention for _check_success.
    """
    if isinstance(node, ast.Call):
        return [node]
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        out: list[ast.Call] = []
        for operand in node.values:
            out.extend(_flatten_and(operand, skill_name))
        return out
    raise ExtractionError(
        f"terminal guard in _check_success_{skill_name} must be a conjunction "
        f"of predicate calls; got {type(node).__name__} at line {node.lineno}",
        skill_name=skill_name,
        phase=3,
        lineno=node.lineno,
    )


# ---------- Phase 4 helpers — reset body → pre ----------

def _extract_pre(
    fn_def: ast.FunctionDef,
    formals: tuple[str, ...],
    skill_name: str,
) -> frozenset[Literal]:
    """Collect all `init_*(obs, ...)` Expr(Call) statements in the reset body."""
    calls: list[ast.Call] = []
    for stmt in fn_def.body:
        # Skip docstrings (Expr(Constant(str))) defensively
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            calls.append(stmt.value)
            continue
        # Anything else (If, Assign, For, While, Return, ...) is out of scope
        raise ExtractionError(
            f"reset_to_init_{skill_name}: body must be a flat sequence of "
            f"initializer calls; got {type(stmt).__name__} at line {stmt.lineno}",
            skill_name=skill_name,
            phase=4,
            lineno=stmt.lineno,
        )
    if not calls:
        raise ExtractionError(
            f"reset_to_init_{skill_name}: body has no initializer calls",
            skill_name=skill_name,
            phase=4,
            lineno=fn_def.lineno,
        )
    return frozenset(
        _resolve_call(call, INITIALIZER_REGISTRY, formals, skill_name, phase=4)
        for call in calls
    )


# ---------- Shared Call → Literal resolution ----------

def _resolve_call(
    call: ast.Call,
    registry: dict[str, PredicateTemplate],
    formals: tuple[str, ...],
    skill_name: str,
    *,
    phase: int,
) -> Literal:
    """Map an ast.Call to a symbolic Literal via the given registry.

    Expected shape: `fn_name(obs, arg0, arg1, ...)` where fn_name is registered
    and each argN is a Name whose .id is the un-prefixed form of one of the
    skill's formal parameters (e.g. `part` maps to formal `?part`).
    """
    # Function target must be a bare Name (reject method calls, attribute
    # access, etc.)
    if not isinstance(call.func, ast.Name):
        raise ExtractionError(
            f"call at line {call.lineno} is not a bare-name function call "
            f"({type(call.func).__name__})",
            skill_name=skill_name,
            phase=phase,
            lineno=call.lineno,
        )
    fn_name = call.func.id
    if fn_name not in registry:
        registry_kind = (
            "predicate" if registry is PREDICATE_REGISTRY else "initializer"
        )
        raise ExtractionError(
            f"{fn_name!r} is not a registered {registry_kind} "
            f"(call at line {call.lineno})",
            skill_name=skill_name,
            phase=phase,
            lineno=call.lineno,
            predicate=fn_name,
        )
    tmpl = registry[fn_name]

    # All calls take `obs` as their first arg by convention; skip it.
    if len(call.args) < 1:
        raise ExtractionError(
            f"call {fn_name!r} at line {call.lineno} takes no arguments; "
            "expected `obs` as the first argument",
            skill_name=skill_name,
            phase=phase,
            lineno=call.lineno,
            predicate=fn_name,
        )
    skill_args = call.args[1:]

    if len(skill_args) != tmpl.arity:
        raise ExtractionError(
            f"call {fn_name!r} at line {call.lineno} has {len(skill_args)} "
            f"arg(s) after obs; template arity is {tmpl.arity}",
            skill_name=skill_name,
            phase=phase,
            lineno=call.lineno,
            predicate=fn_name,
        )

    resolved: list[LiteralArg] = []
    for arg in skill_args:
        if not isinstance(arg, ast.Name):
            raise ExtractionError(
                f"call {fn_name!r} arg at line {arg.lineno} is not a bare "
                f"Name ({type(arg).__name__}); only formal-parameter "
                "references are accepted",
                skill_name=skill_name,
                phase=phase,
                lineno=arg.lineno,
                predicate=fn_name,
            )
        formal = f"?{arg.id}"
        if formal not in formals:
            raise ExtractionError(
                f"call {fn_name!r} arg {arg.id!r} at line {arg.lineno} is "
                f"not one of the skill's formals {formals}",
                skill_name=skill_name,
                phase=phase,
                lineno=arg.lineno,
                predicate=fn_name,
            )
        resolved.append(formal)

    return Literal(tmpl.name, tuple(resolved), negated=False)


# ---------- Phase 5 helper — delete-set derivation ----------

def _derive_delete(
    pre: frozenset[Literal],
    raw_add: frozenset[Literal],
    axioms: tuple[Axiom, ...],
) -> frozenset[Literal]:
    """Return { p ∈ pre : ∃ q ∈ raw_add. axioms ⊨ (q → ¬p) }."""
    delete: set[Literal] = set()
    for p in pre:
        for q in raw_add:
            if entails(q, p.negate(), axioms):
                delete.add(p)
                break
    return frozenset(delete)
