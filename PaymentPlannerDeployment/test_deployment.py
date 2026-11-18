"""Offline acceptance tests: no model API, Langfuse, or embedding downloads."""
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from .settings import Settings
from .agent_core import AgentCore
from .server import create_app
from PaymentPlanner.agent import build_agent
from PaymentPlanner.memory import save_planner_preference, update_planner_preference, PaymentPlannerContext
from agents import Model, ModelResponse, RunContextWrapper, SQLiteSession, Usage
from agents.tool_context import ToolContext
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
from fastapi.testclient import TestClient


class ScriptedModel(Model):
    """Exercise the real SDK runner + tool execution without a provider."""
    def __init__(self):
        self.calls = 0

    async def get_response(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            output = [ResponseFunctionToolCall(type="function_call", call_id="budget-call", name="save_planner_preference",
                arguments=json.dumps({"field": "monthly_budget", "value": "2500"}))]
        else:
            output = [ResponseOutputMessage(type="message", id="answer", role="assistant", status="completed",
                content=[ResponseOutputText(type="output_text", text="Budget saved: 2500 USD.", annotations=[])])]
        return ModelResponse(output=output, usage=Usage(requests=1,input_tokens=10,output_tokens=5,total_tokens=15), response_id=None)

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError("Streaming is not used in this deployment.")
        yield


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = Settings("fixture", "x" * 40, Path(self.tmp.name), "owner", "gpt-4.1-mini", 5, 8, 1200, 60, False, False)
        self.estimate = patch("PaymentPlannerDeployment.agent_core.count_items_tokens", return_value=42)
        self.estimate.start()
        self.addCleanup(self.estimate.stop)
        self.client = TestClient(create_app(self.config))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.headers = {"Authorization": "Bearer " + self.config.access_token}

    def body(self, message="2026-09-15 UBER *TRIP $20.00", thread="demo", request_id=None):
        return {"message": message, "thread_id": thread, "request_id": request_id or str(uuid4())}

    def post(self, body):
        return self.client.post("/chat", json=body, headers=self.headers)

    def test_health_and_page(self):
        self.assertEqual(self.client.get("/healthz").json(), {"status": "ok", "mode": "fixture"})
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("textContent", page.text)
        self.assertNotIn("localStorage", page.text)
        self.assertEqual(page.headers["cache-control"], "no-store")
        self.assertEqual(self.client.get("/docs").status_code, 404)

    def test_authentication(self):
        for path, body in [("/chat", self.body()), ("/reset", {"thread_id": "demo"})]:
            self.assertEqual(self.client.post(path, json=body).status_code, 401)
            self.assertEqual(self.client.post(path, json=body, headers={"Authorization": "Bearer wrong"}).status_code, 401)

    def test_fixture_transaction(self):
        response = self.post(self.body())
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertIn('"category": "Transportation"', data["answer"])
        self.assertIn('"amount": 20.0', data["answer"])
        self.assertEqual(data["usage"]["total_tokens"], 0)
        self.assertEqual(data["estimated_history_and_message_tokens"], 42)
        self.assertFalse(data["remembered"])

    def test_sensitive_before_storage(self):
        for message in ["My password is secret-value", "My PIN is 1234", "4111 1111 1111 1111"]:
            response = self.post(self.body(message))
            self.assertEqual(response.status_code, 400)
            self.assertNotIn("secret-value", response.text)
        self.assertFalse((self.config.data_dir / "sessions.db").exists())
        with sqlite3.connect(self.config.data_dir / "web_requests.db") as db:
            self.assertEqual(db.execute("SELECT count(*) FROM requests").fetchone()[0], 0)

    def test_validation_redacts_input(self):
        for body in [self.body(""), self.body("a"*6001), self.body(thread="../../other"),
                     {**self.body("My password is unique-secret"), "request_id": "not-a-uuid"},
                     {**self.body(), "user_id": "attacker"}]:
            response = self.post(body)
            self.assertEqual(response.status_code, 422)
            self.assertNotIn("unique-secret", response.text)
        self.assertEqual(self.post(self.body("   ")).status_code, 400)

    def test_body_byte_limit(self):
        response = self.client.post("/chat", content=b"x"*40001, headers=self.headers)
        self.assertEqual(response.status_code, 413)

    def test_idempotency_and_intentional_repeat(self):
        body = self.body()
        first = self.post(body).json()
        second = self.post(body).json()
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(first["answer"], second["answer"])
        self.assertEqual(self.post({**body, "message": "NETFLIX $15.00"}).status_code, 409)
        self.assertFalse(self.post(self.body()).json()["cached"])
        session = SQLiteSession(self.client.app.state.core.session_id("demo"), str(self.config.data_dir / "sessions.db"))
        self.assertEqual(len(asyncio.run(session.get_items())), 4)

    def test_cache_survives_restart(self):
        body = self.body()
        self.post(body)
        core = AgentCore(self.config)
        self.assertTrue(asyncio.run(core.chat(body["message"], body["thread_id"], body["request_id"]))["cached"])

    def test_reset_is_scoped(self):
        a, b = self.body(thread="a"), self.body(thread="b")
        self.post(a); self.post(b)
        response = self.client.post("/reset", json={"thread_id": "a"}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(self.post(a).json()["cached"])
        self.assertTrue(self.post(b).json()["cached"])

    def test_rate_limit(self):
        self.client.app.state.settings = replace(self.config, rate_limit=1)
        self.assertEqual(self.post(self.body()).status_code, 200)
        self.assertEqual(self.post(self.body()).status_code, 429)

    def test_tiktoken_failure_is_nonfatal(self):
        with patch("PaymentPlannerDeployment.agent_core.count_items_tokens", side_effect=OSError("offline")):
            response = self.post(self.body())
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["estimated_history_and_message_tokens"])
        self.assertTrue(response.json()["warnings"])

    def fake_result(self):
        return SimpleNamespace(final_output="Your budget is saved.", context_wrapper=SimpleNamespace(
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15)),
            new_items=[SimpleNamespace(raw_item={"type": "function_call", "name": "save_planner_preference"})])

    def test_live_workflow_and_profile_tool(self):
        core = self.client.app.state.core
        core.settings = replace(self.config, mode="openai")
        memory = SimpleNamespace(recall=lambda _: [{"text": "use tables", "topic": "report_style", "kind": "preference"}], remember=unittest.mock.Mock())
        core.memory = memory
        async def runner(agent, message, **kwargs):
            self.assertEqual(agent.model, self.config.model)
            self.assertFalse(kwargs["run_config"].trace_include_sensitive_data)
            arguments = json.dumps({"field": "monthly_budget", "value": "3000"})
            wrapper = ToolContext.from_agent_context(RunContextWrapper(kwargs["context"]), "test-call",
                tool_name="save_planner_preference", tool_arguments=arguments, run_config=kwargs["run_config"])
            result = await save_planner_preference.on_invoke_tool(wrapper, arguments)
            self.assertEqual(result["status"], "ok")
            return self.fake_result()
        with patch("PaymentPlannerDeployment.agent_core.Runner.run", side_effect=runner):
            response = self.post(self.body("My monthly budget is 3000"))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["remembered"])
        self.assertEqual(response.json()["profile"]["monthly_budget"], 3000)
        self.assertEqual(response.json()["usage"]["total_tokens"], 15)
        self.assertEqual(json.loads((self.config.data_dir / "planner_profile.json").read_text())["monthly_budget"], 3000)
        memory.remember.assert_called_once()

    def test_live_error_does_not_leak_and_can_retry(self):
        core = self.client.app.state.core
        core.settings = replace(self.config, mode="openai")
        core.memory = SimpleNamespace(recall=lambda _: [])
        body = self.body()
        with patch("PaymentPlannerDeployment.agent_core.Runner.run", side_effect=RuntimeError("sk-secret-provider-key")):
            response = self.post(body)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("sk-secret", response.text)
        with patch("PaymentPlannerDeployment.agent_core.Runner.run", new=AsyncMock(return_value=self.fake_result())):
            self.assertEqual(self.post(body).status_code, 200)

    def test_real_sdk_runner_tools_and_history(self):
        core = self.client.app.state.core
        core.settings = replace(self.config, mode="openai")
        core.memory = SimpleNamespace(recall=lambda _: [], remember=unittest.mock.Mock())
        model = ScriptedModel()
        def factory(recalled, **kwargs):
            agent = build_agent(recalled, **kwargs)
            agent.model = model
            return agent
        with patch("PaymentPlannerDeployment.agent_core.build_agent", side_effect=factory):
            response = self.post(self.body("My monthly budget is 2500"))
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["profile"]["monthly_budget"], 2500)
        self.assertIn("save_planner_preference", data["tools_used"])
        self.assertEqual(data["usage"]["total_tokens"], 30)
        self.assertEqual(model.calls, 2)
        session = SQLiteSession(core.session_id("demo"), str(self.config.data_dir / "sessions.db"))
        self.assertGreaterEqual(len(asyncio.run(session.get_items())), 4)

    def test_timeout_releases_lock(self):
        core = self.client.app.state.core
        core.settings = replace(self.config, mode="openai", timeout_seconds=0.01)
        core.memory = SimpleNamespace(recall=lambda _: [])
        async def slow(*args, **kwargs):
            await asyncio.sleep(1)
        with patch("PaymentPlannerDeployment.agent_core.Runner.run", side_effect=slow):
            self.assertEqual(self.post(self.body()).status_code, 504)
        self.assertFalse(core.lock.locked())

    def test_agent_has_existing_tools_and_model(self):
        agent = build_agent(model="gpt-4.1-mini", max_output_tokens=1200)
        self.assertEqual(len(agent.tools), 11)
        self.assertEqual(agent.model_settings.max_tokens, 1200)
        self.assertIn("summarize_accounts", [tool.name for tool in agent.tools])

    def test_profile_rejects_nonfinite_values(self):
        for value in ["NaN", "Infinity", "-1"]:
            result = update_planner_preference(PaymentPlannerContext(), "monthly_budget", value, self.config.data_dir / "profile.json")
            self.assertEqual(result["status"], "error")

    def test_configuration_requires_secrets(self):
        with patch.dict(os.environ, {"PAYMENT_PLANNER_ACCESS_TOKEN": "", "PAYMENT_PLANNER_MODE": "fixture"}):
            with self.assertRaises(RuntimeError):
                Settings.from_env()
        with patch.dict(os.environ, {"PAYMENT_PLANNER_ACCESS_TOKEN": "x"*40, "PAYMENT_PLANNER_MODE": "openai", "OPENAI_API_KEY": ""}):
            with self.assertRaises(RuntimeError):
                Settings.from_env()


if __name__ == "__main__":
    unittest.main()
