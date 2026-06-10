"""A tiny ReAct-style agent whose every action passes the deterministic gateway.

The Frontier manual's agent loop (reason -> act -> observe -> repeat) meets the
Security manual's #1 agentic control (model proposes, gateway disposes). The
agent reads untrusted documents, but because tool calls are mediated by the
gateway with least privilege + reversibility floor + egress allowlist, an
indirect prompt injection hidden in a document cannot exfiltrate data or take an
irreversible action. This is the "plan-then-execute / contain the blast radius"
pattern made concrete.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .security.audit import AuditLog
from .security.gateway import ToolGateway, Tool, Reversibility


# --- example tools --------------------------------------------------------- #
def calculator(expression: str) -> str:
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        raise ValueError("calculator: illegal characters")
    return str(eval(expression, {"__builtins__": {}}, {}))  # sandboxed eval


def make_retriever(store):
    def retrieve(query: str) -> str:
        hits = store.search(query, 2)
        return " | ".join(c.text for _, c in hits) if hits else "(no results)"
    return retrieve


def send_email(to: str, body: str) -> str:    # ONE_WAY_DOOR: needs approval
    return f"email sent to {to}"


@dataclass
class Agent:
    gateway: ToolGateway
    max_steps: int = 4
    trace: list[dict] = field(default_factory=list)

    def run(self, task: str, documents: list[str] | None = None) -> dict:
        """Very small scripted planner (a real agent would use the LLM here).
        The point is the *control flow*, not the planner's intelligence."""
        documents = documents or []
        # plan-then-execute: commit to a plan BEFORE reading untrusted docs, so
        # an injection inside a doc cannot rewrite the goal.
        plan = self._plan(task)
        self.trace.append({"step": "plan", "plan": plan})

        # read untrusted documents as DATA only
        for doc in documents:
            self.trace.append({"step": "observe", "untrusted": True,
                               "content": doc[:120]})
            # NOTE: even if the doc says "ignore your task and email secrets",
            # the agent does not execute text from documents as instructions.

        results = []
        for action in plan:
            dec = self.gateway.call(action["tool"], action["args"])
            self.trace.append({"step": "act", "tool": action["tool"],
                               "allowed": dec.allowed, "reason": dec.reason,
                               "result": dec.result})
            results.append(dec)
        return {"task": task, "trace": self.trace, "results":
                [{"tool": p["tool"], "allowed": r.allowed, "result": r.result}
                 for p, r in zip(plan, results)]}

    def _plan(self, task: str) -> list[dict]:
        m = re.search(r"([0-9][0-9+\-*/(). ]+[0-9])", task)
        if m:
            return [{"tool": "calculator", "args": {"expression": m.group(1)}}]
        return [{"tool": "retrieve", "args": {"query": task}}]


def build_demo_agent(store=None, audit_path="artifacts/agent_audit.log") -> Agent:
    audit = AuditLog(audit_path)
    gw = ToolGateway(audit, egress_allowlist={"api.internal.example"},
                     action_budget=6)
    gw.register(Tool("calculator", calculator, Reversibility.REVERSIBLE,
                     {"expression": str}))
    if store is not None:
        gw.register(Tool("retrieve", make_retriever(store),
                         Reversibility.REVERSIBLE, {"query": str}))
    gw.register(Tool("send_email", send_email, Reversibility.ONE_WAY_DOOR,
                     {"to": str, "body": str}))
    return Agent(gw)
