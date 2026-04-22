"""Predicate and initializer registries.

The `@predicate("name(?args)")` and `@initializer("name(?args)")` decorators
record a mapping from a Python function's qualified name to its symbolic
predicate template. The AST extractor uses these registries to resolve
`ast.Call` nodes (e.g. `holding(obs, peg)`) back to symbolic literals
(e.g. `holding(?peg)`).

Design constraints:
  - Registry key is the function's unqualified name. This is sufficient because
    inside reward_specs.py all predicate/initializer calls are module-level
    symbols, never bound methods or locals.
  - Duplicate registration raises VocabularyError. We never silently overwrite.
  - Template parsing is strict: `name` (zero-arg) or `name(?a, ?b, ...)`.
    Other forms raise at decoration time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, TypeVar

from .exceptions import VocabularyError

F = TypeVar("F", bound=Callable)


@dataclass(frozen=True)
class PredicateTemplate:
    """Symbolic predicate schema.

    Attributes:
        name: predicate identifier, e.g. "holding".
        arity: number of formal parameters (0 for propositional predicates).
        formal_params: tuple of formal parameter names in declaration order,
            each starting with "?" (e.g. ("?p",) or ("?peg", "?socket")).
            Empty tuple for 0-arity predicates.
    """

    name: str
    arity: int
    formal_params: tuple[str, ...]


# ---------- Registry storage ----------
PREDICATE_REGISTRY: dict[str, PredicateTemplate] = {}
INITIALIZER_REGISTRY: dict[str, PredicateTemplate] = {}


# ---------- Template parsing ----------
_TEMPLATE_RE = re.compile(
    r"^(?P<name>[a-z_][a-z0-9_]*)(?:\((?P<args>[^)]*)\))?$"
)


def _parse_template(template: str) -> PredicateTemplate:
    m = _TEMPLATE_RE.match(template.strip())
    if m is None:
        raise VocabularyError(
            f"predicate template {template!r} is malformed; "
            "expected 'name' or 'name(?a, ?b, ...)'"
        )
    name = m.group("name")
    args_raw = m.group("args")
    if args_raw is None or args_raw.strip() == "":
        return PredicateTemplate(name=name, arity=0, formal_params=())

    params = tuple(a.strip() for a in args_raw.split(",") if a.strip())
    for p in params:
        if not p.startswith("?") or len(p) < 2:
            raise VocabularyError(
                f"formal parameter {p!r} in template {template!r} must start "
                "with '?' and be at least two characters"
            )
    return PredicateTemplate(name=name, arity=len(params), formal_params=params)


# ---------- Decorators ----------
def predicate(template: str) -> Callable[[F], F]:
    """Register `fn` as the implementation of the symbolic predicate `template`.

    Usage:
        @predicate("holding(?p)")
        def holding(obs, p): ...

    The decorated function's body is irrelevant to the AST extractor; only the
    function's unqualified name is used as the registry key.
    """
    tmpl = _parse_template(template)

    def _decorate(fn: F) -> F:
        key = fn.__name__
        if key in PREDICATE_REGISTRY:
            raise VocabularyError(
                f"predicate function {key!r} already registered as "
                f"{PREDICATE_REGISTRY[key]!r}; duplicate registration of {tmpl!r}"
            )
        PREDICATE_REGISTRY[key] = tmpl
        return fn

    return _decorate


def initializer(template: str) -> Callable[[F], F]:
    """Register `fn` as an initializer establishing the predicate `template`.

    Used for reset_to_init_* functions in reward_specs.py. A call to the
    decorated function (typically `init_on_surface(obs, p)`) is interpreted
    by the AST extractor as asserting the registered predicate as an initial-
    state literal.
    """
    tmpl = _parse_template(template)

    def _decorate(fn: F) -> F:
        key = fn.__name__
        if key in INITIALIZER_REGISTRY:
            raise VocabularyError(
                f"initializer function {key!r} already registered as "
                f"{INITIALIZER_REGISTRY[key]!r}; duplicate registration of {tmpl!r}"
            )
        INITIALIZER_REGISTRY[key] = tmpl
        return fn

    return _decorate


# ---------- Test-only registry reset ----------
def _reset_registries_for_tests() -> None:
    """Clear both registries. Intended for test isolation only.

    Not referenced outside tests; making it public keeps the intent explicit.
    """
    PREDICATE_REGISTRY.clear()
    INITIALIZER_REGISTRY.clear()
