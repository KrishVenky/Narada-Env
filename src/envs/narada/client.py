"""
Narada: Python WebSocket client.

Import NaradaEnv in training code. Never import from server/.

Usage (async):
    async with NaradaEnv(base_url="https://...") as env:
        result = await env.reset(task_type="monogenic")
        result = await env.step(NaradaAction(...))

Usage (sync):
    with NaradaEnv(base_url="...").sync() as env:
        result = env.reset(task_type="monogenic")
        result = env.step(NaradaAction(...))
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from .models import NaradaAction, NaradaState, StepResult

logger = logging.getLogger(__name__)


class NaradaEnv:
    """Async WebSocket client. One instance = one persistent session."""

    def __init__(self, base_url: str = "https://krishvenky-narada-env.hf.space") -> None:
        self._base_url = base_url.rstrip("/")
        self._ws_url = (
            self._base_url
            .replace("https://", "wss://")
            .replace("http://", "ws://")
        ) + "/ws"
        self._ws: Optional[Any] = None

    async def __aenter__(self) -> "NaradaEnv":
        await self._connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._disconnect()

    async def _connect(self) -> None:
        logger.info("Connecting to %s", self._ws_url)
        self._ws = await websockets.connect(
            self._ws_url,
            ping_interval=15,
            ping_timeout=45,
            open_timeout=30,
        )
        logger.info("Connected")

    async def _disconnect(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def reset(
        self,
        task_type: str = "monogenic",
        seed: Optional[int] = None,
    ) -> StepResult:
        msg: Dict[str, Any] = {"type": "reset", "task_type": task_type}
        if seed is not None:
            msg["seed"] = seed
        response = await self._send_recv(msg)
        return self._parse_result(response)

    async def step(self, action: NaradaAction) -> StepResult:
        msg = {"type": "step", "action": action.model_dump()}
        response = await self._send_recv(msg)
        return self._parse_result(response)

    async def state(self) -> NaradaState:
        response = await self._send_recv({"type": "state"})
        if response.get("type") == "error":
            raise RuntimeError(response.get("message", "Server error"))
        return NaradaState.model_validate(response["data"])

    async def _send_recv(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        if self._ws is None:
            await self._connect()
        await self._ws.send(json.dumps(msg))
        raw = await self._ws.recv()
        return json.loads(raw)

    @staticmethod
    def _parse_result(response: Dict[str, Any]) -> StepResult:
        if response.get("type") == "error":
            raise RuntimeError(response.get("message", "Server error"))
        return StepResult.model_validate(response["data"])

    def sync(self) -> "SyncNaradaEnv":
        return SyncNaradaEnv(self)


class SyncNaradaEnv:
    """Synchronous wrapper for use in non-async training loops."""

    def __init__(self, async_env: NaradaEnv) -> None:
        self._env = async_env
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def __enter__(self) -> "SyncNaradaEnv":
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._env._connect())
        return self

    def __exit__(self, *args: Any) -> None:
        if self._loop:
            self._loop.run_until_complete(self._env._disconnect())
            self._loop.close()
            self._loop = None

    def _run(self, coro: Any) -> Any:
        if not self._loop:
            raise RuntimeError("Must be used as a context manager")
        return self._loop.run_until_complete(coro)

    def reset(self, task_type: str = "monogenic", seed: Optional[int] = None) -> StepResult:
        return self._run(self._env.reset(task_type=task_type, seed=seed))

    def step(self, action: NaradaAction) -> StepResult:
        return self._run(self._env.step(action))

    def state(self) -> NaradaState:
        return self._run(self._env.state())
