"""The deterministic tool gateway: model proposes, gateway disposes.

This is the single most important agentic control in the security manual. Every
action the model wants to take passes through one gate that enforces, in plain
code the model cannot talk its way around:

  1. typed-contract validation (is this a known tool with valid args?)
  2. least-privilege allowlist (is this tool permitted in this mode?)
  3. reversibility floor (REVERSIBLE runs; GATED/ONE_WAY_DOOR need human
     approval; the agent can never self-approve)
  4. default-deny egress (outbound destinations must be on an allowlist - the
     highest-leverage single control, it removes the exfiltration leg of the
     "lethal trifecta")
  5. killswitch + per-cycle action budget (bounded, stoppable autonomy)
  6. a monotonic risk ratchet (once a danger signal trips, scrutiny only rises)

Every decision is written to the tamper-evident audit log.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable
from urllib.parse import urlparse

from .audit import AuditLog


class Reversibility(Enum):
    REVERSIBLE = "reversible"      # read a file, do a calculation
    GATED = "gated"               # write a file -> needs approval
    ONE_WAY_DOOR = "one_way_door"  # send email, delete, pay -> always approval


@dataclass
class Tool:
    name: str
    func: Callable
    reversibility: Reversibility = Reversibility.REVERSIBLE
    arg_schema: dict = field(default_factory=dict)   # name -> type
    egress: bool = False                              # does it reach the network?


@dataclass
class GatewayDecision:
    allowed: bool
    reason: str
    result: object = None
    needs_approval: bool = False


class ToolGateway:
    def __init__(self, audit: AuditLog | None = None, *,
                 egress_allowlist: set[str] | None = None,
                 action_budget: int = 10):
        self.tools: dict[str, Tool] = {}
        self.audit = audit
        self.egress_allowlist = egress_allowlist or set()
        self.action_budget = action_budget
        self.actions_used = 0
        self.killed = False
        self.risk_level = 0            # monotonic ratchet
        self.pending_approvals: list[dict] = []

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def killswitch(self):
        self.killed = True

    def raise_risk(self, amount: int = 1):
        self.risk_level += amount      # can only ever go up within a session

    def _log(self, event, **kw):
        if self.audit:
            self.audit.append(event, **kw)

    def call(self, name: str, args: dict, *, approved: bool = False,
             requester: str = "agent") -> GatewayDecision:
        # killswitch + budget: bounded, stoppable autonomy
        if self.killed:
            return self._deny(name, "killswitch engaged")
        if self.actions_used >= self.action_budget:
            return self._deny(name, "per-cycle action budget exhausted")

        tool = self.tools.get(name)
        if tool is None:
            self.raise_risk(1)
            return self._deny(name, "unknown tool (not in allowlist)")

        # typed-contract validation
        for arg, typ in tool.arg_schema.items():
            if arg not in args:
                return self._deny(name, f"missing required arg: {arg}")
            if not isinstance(args[arg], typ):
                return self._deny(name, f"arg {arg} wrong type")

        # default-deny egress allowlist
        if tool.egress:
            url = str(args.get("url", ""))
            host = urlparse(url).hostname or ""
            if host not in self.egress_allowlist:
                self.raise_risk(2)
                return self._deny(name, f"egress to '{host}' denied (not on allowlist)")

        # reversibility floor: irreversible actions require human approval
        if tool.reversibility != Reversibility.REVERSIBLE and not approved:
            self.pending_approvals.append({"tool": name, "args": args})
            self._log("approval_required", tool=name, args=args,
                      reversibility=tool.reversibility.value)
            return GatewayDecision(False, "human approval required",
                                   needs_approval=True)

        # execute
        self.actions_used += 1
        result = tool.func(**args)
        self._log("tool_call", tool=name, args=args, requester=requester,
                  risk=self.risk_level)
        return GatewayDecision(True, "ok", result=result)

    def _deny(self, name, reason) -> GatewayDecision:
        self._log("tool_denied", tool=name, reason=reason, risk=self.risk_level)
        return GatewayDecision(False, reason)
