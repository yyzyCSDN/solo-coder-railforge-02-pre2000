"""Locomotive cycle planning with spot-repair (临修) insertion.

A cycle plan assigns locomotives to scheduled trains. Every assignment is
validated inside a single version boundary against five concerns:

1. tractive effort (牵引能力) -- assigned consist must pull the train;
2. locomotive availability (机车可用时间) -- position, free time, shop blocks;
3. maintenance evidence (维护证据) -- release evidence must be valid at depart;
4. round-trip turnaround (往返周转) -- minimum turn gap and return to home;
5. existing train plans (既有列车计划) -- published commitments are honoured.

Spot repairs never invalidate predecessor cycles silently: callers choose
explicitly between PRESERVE (replan to a still-feasible plan) and ROLLBACK
(reject the insertion, keep the base plan).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from railforge.core.time import utc
from railforge.locomotive.tractive_effort import required_effort
from railforge.maintenance.release import Evidence, ReleasePolicy, releasable

PRESERVE = "preserve"
ROLLBACK = "rollback"
ROLLBACK_MODES = frozenset({PRESERVE, ROLLBACK})


@dataclass(frozen=True)
class ScheduledTrain:
    """A train path that needs powered assignment (既有列车计划中的一个牵引需求)."""
    train_id: str
    origin: str
    destination: str
    depart: datetime
    arrive: datetime
    tonnes: float
    gradient_permille: float
    curve_resistance_kn: float = 0.0


@dataclass(frozen=True)
class Loco:
    unit_id: str
    home: str
    location: str
    available_at: datetime
    continuous_kn: float
    adhesion_kn: float
    due_hours: float = float("inf")


@dataclass(frozen=True)
class ExistingCommitment:
    """A unit already pinned by a previously published plan."""
    unit_id: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class SpotRepair:
    """Unscheduled repair order (临修插单).

    ``releases`` are the maintenance evidence items produced when the repair
    completes; the unit cannot lead a train until fresh evidence covers it.
    """
    repair_id: str
    unit_id: str
    start: datetime
    end: datetime
    releases: tuple[Evidence, ...] = ()


@dataclass(frozen=True)
class Assignment:
    train_id: str
    units: tuple[str, ...]
    origin: str
    destination: str
    depart: datetime
    arrive: datetime


@dataclass(frozen=True)
class CyclePlan:
    version: int
    assignments: tuple[Assignment, ...]
    repairs: tuple[SpotRepair, ...] = ()


@dataclass(frozen=True)
class Violation:
    code: str
    detail: str
    train_id: str | None = None
    unit_id: str | None = None


class InfeasibleCycle(ValueError):
    def __init__(self, violations: list[Violation]):
        self.violations = violations
        super().__init__("; ".join(f"{v.code}:{v.detail}" for v in violations))


@dataclass(frozen=True)
class CycleContext:
    trains: tuple[ScheduledTrain, ...]
    locos: tuple[Loco, ...]
    evidence: dict[str, tuple[Evidence, ...]]
    commitments: tuple[ExistingCommitment, ...] = ()
    required_evidence: frozenset[str] = frozenset()
    turn: timedelta = timedelta(minutes=30)
    adhesion_factor: float = 1.0
    require_return: bool = True


# --------------------------------------------------------------------------- #
# serialisation -- event replay must be able to rebuild plans from events
# --------------------------------------------------------------------------- #

def _dt(value) -> datetime:
    return utc(value) if isinstance(value, datetime) else datetime.fromisoformat(value)


def evidence_to_data(e: Evidence) -> dict:
    return {
        "kind": e.kind,
        "completed_at": utc(e.completed_at).isoformat(),
        "valid_for_seconds": e.valid_for.total_seconds(),
        "passed": e.passed,
    }


def evidence_from_data(row: dict) -> Evidence:
    return Evidence(row["kind"], _dt(row["completed_at"]),
                    timedelta(seconds=row["valid_for_seconds"]), row["passed"])


def repair_to_data(r: SpotRepair) -> dict:
    return {
        "repair_id": r.repair_id,
        "unit_id": r.unit_id,
        "start": utc(r.start).isoformat(),
        "end": utc(r.end).isoformat(),
        "releases": [evidence_to_data(e) for e in r.releases],
    }


def repair_from_data(row: dict) -> SpotRepair:
    return SpotRepair(
        row["repair_id"], row["unit_id"], _dt(row["start"]), _dt(row["end"]),
        tuple(evidence_from_data(e) for e in row.get("releases", ())),
    )


def assignment_to_data(a: Assignment) -> dict:
    return {
        "train_id": a.train_id,
        "units": list(a.units),
        "origin": a.origin,
        "destination": a.destination,
        "depart": utc(a.depart).isoformat(),
        "arrive": utc(a.arrive).isoformat(),
    }


def assignment_from_data(row: dict) -> Assignment:
    return Assignment(row["train_id"], tuple(row["units"]), row["origin"],
                      row["destination"], _dt(row["depart"]), _dt(row["arrive"]))


def plan_to_data(plan: CyclePlan) -> dict:
    return {
        "version": plan.version,
        "assignments": [assignment_to_data(a) for a in plan.assignments],
        "repairs": [repair_to_data(r) for r in plan.repairs],
    }


def plan_from_data(row: dict) -> CyclePlan:
    return CyclePlan(
        row["version"],
        tuple(assignment_from_data(a) for a in row["assignments"]),
        tuple(repair_from_data(r) for r in row.get("repairs", ())),
    )


def violation_to_data(v: Violation) -> dict:
    return {"code": v.code, "detail": v.detail,
            "train_id": v.train_id, "unit_id": v.unit_id}


def violation_from_data(row: dict) -> Violation:
    return Violation(row["code"], row["detail"], row["train_id"], row["unit_id"])


# --------------------------------------------------------------------------- #
# validation -- the single version boundary for all five concerns
# --------------------------------------------------------------------------- #

def _effort(loco: Loco, adhesion_factor: float) -> float:
    return min(loco.continuous_kn, loco.adhesion_kn * adhesion_factor)


def _blocked_for(ctx: CycleContext, repairs: tuple[SpotRepair, ...], unit_id: str):
    rows = [(utc(c.start), utc(c.end)) for c in ctx.commitments if c.unit_id == unit_id]
    rows += [(utc(r.start), utc(r.end)) for r in repairs if r.unit_id == unit_id]
    return sorted(rows)


def _evidence_for(ctx: CycleContext, repairs: tuple[SpotRepair, ...], unit_id: str):
    rows = list(ctx.evidence.get(unit_id, ()))
    for r in repairs:
        if r.unit_id == unit_id:
            rows.extend(r.releases)
    return rows


def _overlaps(start: datetime, end: datetime, blocks) -> bool:
    return any(start < b_end and b_start < end for b_start, b_end in blocks)


def validate_plan(plan: CyclePlan, ctx: CycleContext) -> list[Violation]:
    """Check every assignment against traction, availability, evidence,
    round-trip turnaround and existing commitments. Pure: no side effects."""
    trains = {t.train_id: t for t in ctx.trains}
    locos = {l.unit_id: l for l in ctx.locos}
    repairs = tuple(plan.repairs)
    out: list[Violation] = []

    assigned: dict[str, list[Assignment]] = {}
    for a in plan.assignments:
        if a.train_id not in trains:
            out.append(Violation("unknown_train", f"assignment references unknown train {a.train_id}",
                                 a.train_id))
            continue
        t = trains[a.train_id]
        if (a.origin, a.destination, utc(a.depart), utc(a.arrive)) != \
           (t.origin, t.destination, utc(t.depart), utc(t.arrive)):
            out.append(Violation("train_mismatch",
                                 f"assignment for {a.train_id} diverges from published train path",
                                 a.train_id))
        if not a.units:
            out.append(Violation("empty_assignment", "no units assigned", a.train_id))
        for uid in a.units:
            assigned.setdefault(uid, []).append(a)

    missing = sorted(set(trains) - {a.train_id for a in plan.assignments})
    for tid in missing:
        out.append(Violation("uncovered_train", "scheduled train has no powered assignment", tid))

    for uid, chain in assigned.items():
        loco = locos.get(uid)
        if loco is None:
            for a in chain:
                out.append(Violation("unknown_unit", f"unit {uid} not in fleet", a.train_id, uid))
            continue
        chain = sorted(chain, key=lambda a: (utc(a.depart), a.train_id))
        cursor_loc = loco.location
        cursor_free = utc(loco.available_at)
        hours_used = 0.0
        blocks = _blocked_for(ctx, repairs, uid)
        evidence_rows = _evidence_for(ctx, repairs, uid)
        for prev, a in zip([None, *chain[:-1]], chain):
            depart, arrive = utc(a.depart), utc(a.arrive)
            if a.origin != cursor_loc:
                out.append(Violation(
                    "chain_broken",
                    f"unit {uid} is at {cursor_loc} when {a.train_id} departs {a.origin}",
                    a.train_id, uid))
            if depart < cursor_free:
                out.append(Violation(
                    "turnaround",
                    f"unit {uid} not free until {cursor_free.isoformat()}, "
                    f"train departs {depart.isoformat()}",
                    a.train_id, uid))
            if _overlaps(depart, arrive, blocks):
                out.append(Violation(
                    "unavailable",
                    f"unit {uid} blocked by commitment or repair during {a.train_id}",
                    a.train_id, uid))
            if not releasable(evidence_rows, ReleasePolicy(ctx.required_evidence), depart):
                out.append(Violation(
                    "evidence",
                    f"unit {uid} lacks valid maintenance evidence {sorted(ctx.required_evidence)}",
                    a.train_id, uid))
            t = trains.get(a.train_id)
            if t is not None:
                leg_hours = (arrive - depart).total_seconds() / 3600
                hours_used += leg_hours
                if hours_used > loco.due_hours:
                    out.append(Violation(
                        "service_hours",
                        f"unit {uid} exceeds {loco.due_hours} service hours", a.train_id, uid))
            cursor_loc = a.destination
            cursor_free = arrive + ctx.turn
        if ctx.require_return and chain and cursor_loc != loco.home:
            out.append(Violation(
                "return_open",
                f"unit {uid} round trip ends at {cursor_loc}, home is {loco.home}",
                chain[-1].train_id, uid))

    for a in plan.assignments:
        t = trains.get(a.train_id)
        if t is None:
            continue
        power = 0.0
        for uid in a.units:
            loco = locos.get(uid)
            if loco is not None:
                power += _effort(loco, ctx.adhesion_factor)
        need = required_effort(t.tonnes, t.gradient_permille, t.curve_resistance_kn)
        if power + 1e-9 < need:
            out.append(Violation(
                "traction",
                f"train {a.train_id}: effort {power:.1f}kN below required {need:.1f}kN",
                a.train_id))

    return sorted(out, key=lambda v: (v.train_id or "", v.unit_id or "", v.code, v.detail))


# --------------------------------------------------------------------------- #
# plan construction
# --------------------------------------------------------------------------- #

def _chain_state(ctx: CycleContext, repairs: tuple[SpotRepair, ...]):
    state: dict[str, list] = {}
    for loco in ctx.locos:
        state[loco.unit_id] = {
            "loc": loco.location,
            "free": utc(loco.available_at),
            "hours": 0.0,
            "blocks": _blocked_for(ctx, repairs, loco.unit_id),
            "evidence": _evidence_for(ctx, repairs, loco.unit_id),
        }
    return state


def build_cycle(ctx: CycleContext, repairs: tuple[SpotRepair, ...] = (),
                version: int = 0) -> CyclePlan:
    """Greedily assign units in train depart order, then validate the whole
    plan as one boundary. Raises InfeasibleCycle on any violation."""
    state = _chain_state(ctx, repairs)
    locos = {l.unit_id: l for l in ctx.locos}
    assignments: list[Assignment] = []

    for t in sorted(ctx.trains, key=lambda x: (utc(x.depart), x.train_id)):
        depart, arrive = utc(t.depart), utc(t.arrive)
        if arrive <= depart:
            raise InfeasibleCycle([Violation("bad_train", "arrive before depart", t.train_id)])
        leg_hours = (arrive - depart).total_seconds() / 3600
        candidates = []
        for uid, st in state.items():
            loco = locos[uid]
            if st["loc"] != t.origin or depart < st["free"]:
                continue
            if st["hours"] + leg_hours > loco.due_hours:
                continue
            if _overlaps(depart, arrive, st["blocks"]):
                continue
            if not releasable(st["evidence"], ReleasePolicy(ctx.required_evidence), depart):
                continue
            candidates.append(uid)
        candidates.sort(key=lambda uid: (
            state[uid]["free"], -_effort(locos[uid], ctx.adhesion_factor), uid))
        need = required_effort(t.tonnes, t.gradient_permille, t.curve_resistance_kn)
        chosen: list[str] = []
        power = 0.0
        for uid in candidates:
            if power + 1e-9 >= need:
                break
            chosen.append(uid)
            power += _effort(locos[uid], ctx.adhesion_factor)
        if power + 1e-9 < need:
            raise InfeasibleCycle([Violation(
                "locomotive_shortage",
                f"train {t.train_id} needs {need:.1f}kN, only {power:.1f}kN available",
                t.train_id)])
        assignments.append(Assignment(t.train_id, tuple(chosen), t.origin,
                                      t.destination, depart, arrive))
        for uid in chosen:
            st = state[uid]
            st["loc"] = t.destination
            st["free"] = arrive + ctx.turn
            st["hours"] += leg_hours

    plan = CyclePlan(version, tuple(assignments), tuple(repairs))
    violations = validate_plan(plan, ctx)
    if violations:
        raise InfeasibleCycle(violations)
    return plan


# --------------------------------------------------------------------------- #
# spot repair insertion
# --------------------------------------------------------------------------- #

def affected_trains(base: CyclePlan, candidate: CyclePlan) -> list[str]:
    """Train ids whose powered assignment changed -- the predecessor cycles
    the insertion touched (never silently invalidated)."""
    old = {a.train_id: a.units for a in base.assignments}
    new = {a.train_id: a.units for a in candidate.assignments}
    return sorted(tid for tid in set(old) | set(new) if old.get(tid) != new.get(tid))


def insert_repair(base: CyclePlan, ctx: CycleContext, repair: SpotRepair, mode: str):
    """Insert a spot repair against a base plan.

    PRESERVE: replan the whole boundary with the shop block; return the new
    feasible plan (or raise InfeasibleCycle). Prior cycles are revalidated.

    ROLLBACK: keep the base plan untouched whenever the repair conflicts with
    it; return (None, conflicts). A non-conflicting repair is recorded as a
    shop block with assignments unchanged.
    """
    if mode not in ROLLBACK_MODES:
        raise ValueError(f"unknown conflict mode {mode!r}; choose preserve or rollback")
    repairs = tuple(plan.repairs) if (plan := base) else ()
    if any(r.repair_id == repair.repair_id for r in repairs):
        return base, []
    with_repair = repairs + (repair,)

    if mode == PRESERVE:
        candidate = build_cycle(ctx, with_repair, version=base.version + 1)
        return candidate, []

    conflicts = validate_plan(replace(base, repairs=with_repair), ctx)
    if conflicts:
        return None, conflicts
    return replace(base, repairs=with_repair), []
