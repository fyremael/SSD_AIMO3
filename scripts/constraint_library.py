from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ArithmeticCheck:
    expression: str
    lhs: int
    rhs: int
    operator: str
    expected: int
    passed: bool


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def count_lines(text: str) -> int:
    stripped = normalize_text(text).strip()
    if not stripped:
        return 0
    return len([line for line in stripped.split("\n") if line.strip()])


def estimate_token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def has_boxed_answer(text: str) -> bool:
    return bool(re.search(r"\\boxed\{\s*[0-9]{1,10}\s*\}", text))


def has_final_answer_phrase(text: str) -> bool:
    return bool(re.search(r"(?i)final\s+answer\s*[:=]\s*[0-9]{1,10}", text))


_ARITHMETIC_PATTERN = re.compile(
    r"(?<!\d)(-?\d{1,6})\s*([+\-*/])\s*(-?\d{1,6})\s*=\s*(-?\d{1,12})(?!\d)"
)


def check_explicit_integer_arithmetic(text: str) -> List[ArithmeticCheck]:
    checks: List[ArithmeticCheck] = []
    for match in _ARITHMETIC_PATTERN.finditer(text):
        lhs = int(match.group(1))
        operator = match.group(2)
        rhs = int(match.group(3))
        stated = int(match.group(4))
        expected: Optional[int]
        if operator == "+":
            expected = lhs + rhs
        elif operator == "-":
            expected = lhs - rhs
        elif operator == "*":
            expected = lhs * rhs
        elif operator == "/":
            if rhs == 0 or lhs % rhs != 0:
                expected = None
            else:
                expected = lhs // rhs
        else:
            expected = None
        passed = expected is not None and expected == stated
        checks.append(
            ArithmeticCheck(
                expression=match.group(0),
                lhs=lhs,
                rhs=rhs,
                operator=operator,
                expected=expected if expected is not None else math.inf,
                passed=passed,
            )
        )
    return checks


_MODULAR_PATTERN = re.compile(
    r"(?<!\d)(-?\d{1,8})\s*(?:≡|==)\s*(-?\d{1,8})\s*\(\s*mod\s+(-?\d{1,5})\s*\)",
    flags=re.IGNORECASE,
)


def check_explicit_modular_claims(text: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for match in _MODULAR_PATTERN.finditer(text):
        left = int(match.group(1))
        right = int(match.group(2))
        modulus = int(match.group(3))
        if modulus == 0:
            passed = False
        else:
            passed = (left - right) % abs(modulus) == 0
        results.append(
            {
                "claim": match.group(0),
                "left": left,
                "right": right,
                "modulus": modulus,
                "passed": passed,
            }
        )
    return results


def detect_multiple_final_answers(text: str, patterns: Sequence[str]) -> Dict[str, Any]:
    found: List[int] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            try:
                found.append(int(match.group(1)))
            except Exception:
                continue
    unique = sorted(set(found))
    return {
        "num_mentions": len(found),
        "answers": unique,
        "has_conflict": len(unique) > 1,
    }


_PARITY_EVEN_PATTERN = re.compile(r"(?i)\b([a-z])\s+is\s+even\b")
_PARITY_ODD_PATTERN = re.compile(r"(?i)\b([a-z])\s+is\s+odd\b")


def detect_parity_conflicts(text: str) -> Dict[str, Any]:
    even_vars = {match.group(1).lower() for match in _PARITY_EVEN_PATTERN.finditer(text)}
    odd_vars = {match.group(1).lower() for match in _PARITY_ODD_PATTERN.finditer(text)}
    conflicted = sorted(even_vars & odd_vars)
    return {
        "even_vars": sorted(even_vars),
        "odd_vars": sorted(odd_vars),
        "conflicted_vars": conflicted,
        "has_conflict": bool(conflicted),
    }


def build_surface_features(text: str) -> Dict[str, Any]:
    return {
        "num_lines": count_lines(text),
        "token_estimate": estimate_token_count(text),
        "num_equations": len(re.findall(r"=", text)),
        "has_boxed_answer": has_boxed_answer(text),
        "has_final_answer_phrase": has_final_answer_phrase(text),
        "char_length": len(text),
    }


def evaluate_trace_constraints(
    text: str,
    *,
    extracted_answer: Optional[int],
    extraction_status: str,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    contradiction_patterns = config.get("contradiction_patterns", [])
    enabled_checks = config.get("enabled_checks", {})

    features = build_surface_features(text)
    arithmetic_checks = check_explicit_integer_arithmetic(text) if enabled_checks.get("arithmetic_equalities", True) else []
    modular_checks = check_explicit_modular_claims(text) if enabled_checks.get("modular_claims", True) else []
    repeated_finals = (
        detect_multiple_final_answers(text, contradiction_patterns)
        if enabled_checks.get("repeated_final_answer_conflicts", True)
        else {"num_mentions": 0, "answers": [], "has_conflict": False}
    )
    parity = (
        detect_parity_conflicts(text)
        if enabled_checks.get("parity_conflicts", True)
        else {"even_vars": [], "odd_vars": [], "conflicted_vars": [], "has_conflict": False}
    )

    contradiction_count = int(repeated_finals.get("has_conflict", False)) + int(parity.get("has_conflict", False))
    arithmetic_failures = sum(1 for item in arithmetic_checks if not item.passed)
    modular_failures = sum(1 for item in modular_checks if not bool(item.get("passed")))

    notes: List[str] = []
    if repeated_finals.get("has_conflict"):
        notes.append(f"Conflicting final answers detected: {repeated_finals.get('answers')}")
    if parity.get("has_conflict"):
        notes.append(f"Parity conflict on variables: {parity.get('conflicted_vars')}")
    if arithmetic_failures:
        notes.append(f"Arithmetic failures: {arithmetic_failures}")
    if modular_failures:
        notes.append(f"Modular failures: {modular_failures}")
    if extracted_answer is None:
        notes.append("No extracted final answer")

    return {
        "features": features,
        "constraint_checks": {
            "extraction_status": extraction_status,
            "num_arithmetic_failures": arithmetic_failures,
            "num_contradictions": contradiction_count,
            "num_modular_claims_checked": len(modular_checks),
            "num_modular_failures": modular_failures,
            "repeated_final_answers": repeated_finals,
            "parity": parity,
            "notes": notes,
        },
    }


def compute_tropical_penalty(
    features: Mapping[str, Any],
    constraint_checks: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, float]:
    extraction_penalty = 0.0 if constraint_checks.get("extraction_status") == "ok" else float(config.get("extraction_failure_penalty", 100.0))
    contradiction_penalty = float(constraint_checks.get("num_contradictions", 0)) * float(config.get("contradiction_penalty", 8.0))
    arithmetic_penalty = float(constraint_checks.get("num_arithmetic_failures", 0)) * float(config.get("arithmetic_failure_penalty", 4.0))
    arithmetic_penalty += float(constraint_checks.get("num_modular_failures", 0)) * float(config.get("modular_failure_penalty", 4.0))

    structure_penalty = 0.0
    if not bool(features.get("has_boxed_answer")):
        structure_penalty += float(config.get("missing_structure_penalty", 1.5))
    else:
        structure_penalty += float(config.get("reward_boxed_answer", -0.5))
    if not bool(features.get("has_final_answer_phrase")):
        structure_penalty += float(config.get("missing_structure_penalty", 1.5))
    else:
        structure_penalty += float(config.get("reward_final_answer_phrase", -0.5))

    char_length = int(features.get("char_length", 0))
    threshold = int(config.get("overlong_threshold_chars", 1600))
    complexity_penalty = 0.0
    if char_length > threshold:
        excess = char_length - threshold
        complexity_penalty = (excess / 200.0) * float(config.get("complexity_penalty_per_200_chars", 0.5))

    total_penalty = extraction_penalty + contradiction_penalty + arithmetic_penalty + structure_penalty + complexity_penalty
    return {
        "total_penalty": total_penalty,
        "extraction_penalty": extraction_penalty,
        "contradiction_penalty": contradiction_penalty,
        "arithmetic_penalty": arithmetic_penalty,
        "structure_penalty": structure_penalty,
        "complexity_penalty": complexity_penalty,
    }
