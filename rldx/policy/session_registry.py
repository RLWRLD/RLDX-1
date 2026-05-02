# SPDX-License-Identifier: Apache-2.0
"""
SessionRegistry — single owner of per-session mutable state for RLDXPolicy.

Until now, session state lived in three separate places:
  - RLDXPolicy._memory_cache (dict[sid, Tensor])
  - RLDXPolicy._rtc_chunk_cache (dict[sid, Tensor])
  - model._cached_mq (attr, written by RLDXPolicy externally)

This split made two problems structural:
  - reset signal fragmentation — one options flag has to touch three places
  - owner-less state — nobody was responsible for the lifecycle

SessionRegistry collapses this into one container:
  - SessionState is the value type (memory_tokens + rtc_chunk)
  - ResetScope controls what reset means:
      EPISODE  — drop entry entirely (matches BASE `_memory_cache.pop(sid)`)
      RTC_ONLY — keep entry + memory_tokens, clear rtc_chunk only

Lifecycle is **caller-managed**: the registry does not auto-evict. Callers
drop sessions when robots disconnect (via ``drop`` when added, or via
``reset`` / ``clear``). A bound would either silently drop live robot
state (LRU) or refuse new sessions (fail-fast) — neither fits a
safety-critical inference path well enough to make a default.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Iterator


if TYPE_CHECKING:
    import torch


class ResetScope(Enum):
    """What does 'reset' mean for a given sid.

    EPISODE   — session ended; drop the entry entirely (pop semantics,
                matches BASE `_memory_cache.pop(sid)` behavior).
    RTC_ONLY  — partial invalidation; keep the entry and memory_tokens,
                clear rtc_chunk only (e.g. RTC resync without losing
                temporal memory).
    """

    EPISODE = "episode"
    RTC_ONLY = "rtc_only"


@dataclass
class SessionState:
    """Per-session mutable state."""

    memory_tokens: "torch.Tensor | None" = None
    rtc_chunk: "torch.Tensor | None" = None


class SessionRegistry:
    """
    Single-owner container of SessionState keyed by session id.

    Invariants:
      - get_or_create always returns a SessionState; never None
      - reset(sids, scope) is a no-op for sids not in the registry
        (caller may send reset signal before sid has been seen —
        e.g. episode boundary on first step)
      - No auto-eviction — lifetime is caller-managed.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    # ---- lookup ---------------------------------------------------------

    def get_or_create(self, sid: str) -> SessionState:
        """Return state for sid, creating empty state if unseen."""
        if sid not in self._sessions:
            self._sessions[sid] = SessionState()
        return self._sessions[sid]

    def peek(self, sid: str) -> SessionState | None:
        """Return state for sid WITHOUT creating. No side effects."""
        return self._sessions.get(sid)

    def items(self) -> list[tuple[str, SessionState]]:
        """Return (sid, state) pairs in insertion order. Read-only view."""
        return list(self._sessions.items())

    def update(self, sid: str, state: SessionState) -> None:
        """Overwrite state for sid (wholesale). Creates if absent.

        Prefer :meth:`set` for per-field writes — this method is for test
        harnesses and rare cases where caller constructs a SessionState
        externally.
        """
        self._sessions[sid] = state

    def set(self, sid: str, **fields) -> SessionState:
        """Write one or more fields on session state. Creates if absent.
        Returns the final state for convenience.

        Encapsulates the "create-if-absent + mutate" pattern so callers
        never depend on SessionState being a mutable reference.

        Example:
            registry.set(sid, memory_tokens=new_mem)
            registry.set(sid, rtc_chunk=chunk, memory_tokens=mem)  # atomic
        """
        state = self._sessions.get(sid)
        if state is None:
            state = SessionState()
            self._sessions[sid] = state
        for k, v in fields.items():
            if not hasattr(state, k):
                raise AttributeError(
                    f"SessionState has no field {k!r} "
                    f"(known: {list(state.__dataclass_fields__.keys())})"
                )
            setattr(state, k, v)
        return state

    # ---- reset / drop ---------------------------------------------------

    def reset(self, sids: Iterable[str], scope: ResetScope) -> list[str]:
        """Apply `scope` reset to each sid in `sids`.

        - EPISODE: drop entry entirely (equivalent to :meth:`drop`). Matches
          BASE pop-on-reset semantics.
        - RTC_ONLY: keep entry, clear rtc_chunk field only.

        Returns the list of sids that were actually touched (present in
        the registry). Unknown sids are silently skipped.
        """
        touched: list[str] = []
        for sid in sids:
            if sid not in self._sessions:
                continue
            if scope == ResetScope.EPISODE:
                del self._sessions[sid]
            elif scope == ResetScope.RTC_ONLY:
                self._sessions[sid].rtc_chunk = None
            else:  # defensive — unreachable given enum
                raise ValueError(f"unknown ResetScope: {scope}")
            touched.append(sid)
        return touched

    def drop(self, sid: str) -> bool:
        """Remove sid entirely from registry. Returns True if sid was present.

        Intended for explicit session end — e.g. when a robot disconnects
        or an env rollout finishes. Equivalent to ``reset([sid], EPISODE)``
        for a single sid; this method is the convenience form with a boolean
        "was it there?" return value instead of a touched-list.
        """
        return self._sessions.pop(sid, None) is not None

    def clear(self) -> list[str]:
        """Remove all sessions. Returns the list of cleared sids."""
        cleared = list(self._sessions.keys())
        self._sessions.clear()
        return cleared

    # ---- batch ops (session domain service) ---------------------------
    # These encapsulate the per-sid → (B, ...) tensor choreography that
    # RLDXPolicy needs for memory / RTC. Kept here (not in the caller)
    # so session semantics live next to the state, and downstream
    # consumers (PolicyRuntime, ReplayPolicy-like substitutes) get the
    # whole session behavior from one object rather than re-deriving it.

    def resolve_sids(
        self,
        session_ids: list[str] | None,
        batch_size: int,
    ) -> list[str]:
        """Resolve per-slot sids for a B-sized batch.

        If ``session_ids`` is provided with the correct length, it is used
        verbatim (stringified). Otherwise falls back to synthetic per-slot
        ids ``["default_0", "default_1", ...]`` — matches BASE behavior for
        unspecified multi-session inputs.

        The naming convention is a session-domain concern; keeping it here
        means callers don't need to know ``"default_{i}"`` exists.
        """
        if session_ids is not None and len(session_ids) == batch_size:
            return [str(s) for s in session_ids]
        return [f"default_{i}" for i in range(batch_size)]

    def invalidate_rtc(
        self,
        active_sids: list[str],
        reset_memory: "torch.Tensor | None",
    ) -> None:
        """Drop rtc_chunk for each sid where reset_memory is True.

        ``reset_memory`` is the raw bool tensor the caller received (via
        options["reset_memory"]); ``None`` means "no resets". The tensor
        → list conversion and mask filtering live here, not in the caller.
        """
        if reset_memory is None:
            return
        mask = reset_memory.cpu().tolist()
        to_reset = [sid for sid, flag in zip(active_sids, mask) if flag]
        if to_reset:
            self.reset(to_reset, ResetScope.RTC_ONLY)

    def load_memory_batch(
        self,
        sids: list[str],
        reset_mask: list[bool] | None = None,
    ) -> tuple["torch.Tensor | None", list[bool]]:
        """Load per-sid memory_tokens into a batched tensor.

        For each sid:
          - If reset_mask[i] is True: drop the entry (EPISODE semantics) and
            contribute None to the slot.
          - Else: read memory_tokens (None if sid absent).

        Returns:
            stacked: (B, *mem_shape) with zero placeholders in None slots,
                     or None if every slot is None (cold start across batch).
            cold_start_mask: list[bool] of length B, True where the slot had
                             no cached memory. Caller typically uses this to
                             set the model's per-sample reset_memory flag.
        """
        import torch

        cached: list["torch.Tensor | None"] = []
        for idx, sid in enumerate(sids):
            if reset_mask is not None and reset_mask[idx]:
                self.drop(sid)
                cached.append(None)
            else:
                state = self.peek(sid)
                cached.append(state.memory_tokens if state is not None else None)

        cold_start = [c is None for c in cached]

        ref = next((c for c in cached if c is not None), None)
        if ref is None:
            return None, cold_start

        placeholder = torch.zeros_like(ref)
        stacked = torch.stack(
            [c if c is not None else placeholder for c in cached],
            dim=0,
        )
        return stacked, cold_start

    def save_memory_batch(
        self,
        sids: list[str],
        stacked: "torch.Tensor",
    ) -> None:
        """Save per-sid memory_tokens from a batched tensor.

        stacked: (B, *mem_shape) where B == len(sids). Each slot is detached
        and cloned before storing so the model can free its scratchpad.
        """
        for idx, sid in enumerate(sids):
            self.set(sid, memory_tokens=stacked[idx].detach().clone())

    def load_rtc_prefix(
        self,
        sids: list[str],
        delay: int,
        exec_horizon: int,
    ) -> "torch.Tensor | None":
        """Build RTC prefix tensor by slicing cached rtc_chunks.

        For each sid, read rtc_chunk and take ``[exec_horizon : exec_horizon+delay]``.
        Returns stacked tensor of shape (B, delay, D), or None if any sid
        lacks a sufficiently long chunk (cold start / insufficient history).
        """
        import torch

        chunks: list["torch.Tensor | None"] = []
        for sid in sids:
            state = self.peek(sid)
            chunks.append(state.rtc_chunk if state is not None else None)

        needed = exec_horizon + delay
        if not all(c is not None and c.shape[0] >= needed for c in chunks):
            return None

        return torch.stack(
            [c[exec_horizon : exec_horizon + delay] for c in chunks],
            dim=0,
        )

    def load_rtc_postfix_target(
        self,
        sids: list[str],
        delay: int,
        exec_horizon: int,
    ) -> "torch.Tensor | None":
        """Build the new chunk's Y[d:H] postfix target by slicing previous-chunk
        predictions out of the cache (Eq. 5 of arXiv 2506.07339).

        At decision step ``t``, the previous chunk's predictions span absolute
        time positions ``[t-s, t-s+H_prev)``. The new chunk spans absolute
        positions ``[t, t+H)``. Their overlap maps new-chunk position ``i``
        to previous-chunk position ``s + i``, so:

            Y_new[d : H] = previous_chunk[s + d : s + H]

        We slice ``[s + d : ]`` of the cache (clipped to its actual length).
        Positions falling beyond the previous chunk's horizon are absent —
        they'd land in the **free** region of the new chunk where the
        soft-mask ``W = 0`` and Y is irrelevant, so the caller can pad with
        zeros (or any value) without affecting the guidance.

        Returns stacked tensor of shape ``(B, L, D)`` where
        ``L = max(0, cache_len - (s + d))``, or ``None`` if any sid is cold.
        """
        import torch

        chunks: list["torch.Tensor | None"] = []
        for sid in sids:
            state = self.peek(sid)
            chunks.append(state.rtc_chunk if state is not None else None)

        needed = exec_horizon + delay + 1  # need at least one ramp position
        if not all(c is not None and c.shape[0] >= needed for c in chunks):
            return None

        start = exec_horizon + delay
        target_len = min(c.shape[0] - start for c in chunks)
        return torch.stack(
            [c[start : start + target_len] for c in chunks],
            dim=0,
        )

    def save_rtc_batch(
        self,
        sids: list[str],
        normalized_action: "torch.Tensor",
    ) -> None:
        """Save per-sid rtc_chunk from a batched action tensor.

        normalized_action: (B, H, D). Detached + moved to CPU so GPU memory
        isn't pinned across inference calls.
        """
        detached = normalized_action.detach().cpu()
        for idx, sid in enumerate(sids):
            self.set(sid, rtc_chunk=detached[idx].clone())

    @contextmanager
    def memory_scratchpad(
        self,
        model,
        session_ids: list[str] | None,
        batch_size: int,
        reset_memory: "torch.Tensor | None",
    ) -> "Iterator[list[bool] | None]":
        """Manage ``model._cached_mq`` as a per-call scratchpad.

        On enter:  load per-sid memory from registry → inject into ``model._cached_mq``.
        Yield:     cold_start mask (multi-session path) or None (single default fallback).
        On exit:   extract mutated ``model._cached_mq`` → save to registry → clear.

        Multi-session path (session_ids matches batch_size):
            Uses load_memory_batch / save_memory_batch. Yields cold_start: list[bool]
            so caller can set model's reset_memory flag per sample.

        Single-default path (session_ids missing or wrong length):
            Uses "default" sid as a shared session across the batch. Yields None
            (caller has no per-sample cold-start signal to apply). Matches BASE
            behavior for unspecified session inputs.

        ``model._cached_mq`` write/read is owned by this registry method —
        callers should never touch it directly. If ``model`` has no
        ``_cached_mq`` attribute, the context manager is a no-op (memory
        path is simply not engaged).
        """
        is_multi = session_ids is not None and len(session_ids) == batch_size

        if is_multi:
            sids_list = list(session_ids)  # type: ignore[arg-type]
            reset_mask = (
                reset_memory.cpu().tolist() if reset_memory is not None else [False] * batch_size
            )
            stacked, cold_start = self.load_memory_batch(sids_list, reset_mask)
            model._cached_mq = stacked
        else:
            state = self.peek("default")
            cached = state.memory_tokens if state is not None else None
            if reset_memory is not None and reset_memory.all():
                self.drop("default")
                cached = None
            model._cached_mq = cached
            cold_start = None

        try:
            yield cold_start
        finally:
            new_ctx = getattr(model, "_cached_mq", None)
            if new_ctx is not None:
                if is_multi:
                    self.save_memory_batch(sids_list, new_ctx)
                else:
                    self.set("default", memory_tokens=new_ctx.detach().clone())
            model._cached_mq = None

    # ---- introspection --------------------------------------------------

    def __len__(self) -> int:
        return len(self._sessions)

    def __contains__(self, sid: str) -> bool:
        return sid in self._sessions

    def sids(self) -> list[str]:
        """Return current sids in insertion order."""
        return list(self._sessions.keys())
