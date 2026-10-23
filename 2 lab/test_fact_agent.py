import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from demo_restart import demo_chains, run_demo
from fact_agent import (
    EXTRACTION_PROMPT, FactAgent, FactMemory, Facts, MemoryFileError,
    build_live_chains, lab1, make_reply_prompt,
)


class FactAgentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "memory.json"
        self.memory = FactMemory(self.path, "alice")

    def test_missing_file_and_financial_greeting(self):
        self.assertEqual(self.memory.load(), Facts())
        extraction, response = demo_chains([{}], ["Здравствуйте"])
        self.assertIn("финансовыми", FactAgent(self.memory, extraction, response).greeting())
        self.assertFalse(self.path.exists())

    def test_corrections_nulls_and_profile_isolation(self):
        self.memory.update(Facts(name="Анна", stack=["Python"], level="junior"))
        other = FactMemory(self.path, "bob")
        other.update(Facts(name="Борис"))
        self.memory.update(Facts(level="middle", stack=[]))
        reread = FactMemory(self.path, "alice").load()
        self.assertEqual(reread.level, "middle")
        self.assertEqual(reread.name, "Анна")
        self.assertEqual(reread.stack, ["Python"])
        self.assertEqual(other.load().name, "Борис")
        self.assertIsNone(other.load().level)

    def test_corrupt_and_wrong_schema_preserved(self):
        for content in ('{bad', '{"version": 2}', '{"profiles": {"alice": {"name": 123}}}'):
            self.path.write_text(content, encoding="utf-8")
            with self.assertRaises(MemoryFileError):
                FactMemory(self.path, "alice")
            self.assertEqual(self.path.read_text(encoding="utf-8"), content)

    def test_invalid_extraction_does_not_overwrite(self):
        self.memory.update(Facts(name="Анна"))
        before = self.path.read_bytes()
        agent = FactAgent(self.memory, RunnableLambda(lambda _: {"name": 42}),
                          RunnableLambda(lambda _: "unused"))
        with self.assertRaises(ValidationError):
            agent.send("Привет")
        self.assertEqual(self.path.read_bytes(), before)

    def test_failed_atomic_replace_preserves_previous_file(self):
        self.memory.update(Facts(name="Анна"))
        before = self.path.read_bytes()
        with patch("fact_agent.os.replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                self.memory.update(Facts(level="middle"))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_response_failure_still_saves_facts(self):
        def fail(_):
            raise ConnectionError("offline")
        agent = FactAgent(self.memory, RunnableLambda(lambda _: Facts(name="Анна")), RunnableLambda(fail))
        with self.assertRaises(ConnectionError):
            agent.send("Меня зовут Анна")
        self.assertEqual(FactMemory(self.path, "alice").load().name, "Анна")

    def test_exit_and_empty_input_do_not_call_model(self):
        def unexpected(_):
            self.fail("Model must not be called")
        agent = FactAgent(self.memory, RunnableLambda(unexpected), RunnableLambda(unexpected))
        with self.assertRaises(ValueError):
            agent.send(" ")
        agent.send("/exit")
        self.assertTrue(agent.closed)
        self.assertFalse(self.path.exists())
        with self.assertRaises(RuntimeError):
            agent.send("hello")

    def test_question_context_and_saved_facts_reach_chains(self):
        extraction_inputs, response_inputs, debug = [], [], []
        def extract(data):
            extraction_inputs.append(data)
            return Facts(name="Анна")
        def respond(data):
            response_inputs.append(data)
            return "Рада знакомству."
        agent = FactAgent(self.memory, RunnableLambda(extract), RunnableLambda(respond), debug.append)
        agent.last_reply = "Как вас зовут?"
        reply = agent.send("Анна")
        self.assertEqual(extraction_inputs[0]["question"], "Как вас зовут?")
        self.assertEqual(json.loads(response_inputs[0]["facts"])["name"], "Анна")
        self.assertEqual(reply, "Рада знакомству.")
        self.assertNotIn("Как вас зовут?", reply)
        self.assertEqual(debug[-1]["event"], "facts_saved")
        self.assertNotIn("facts_saved", reply)
        stored = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(set(stored), {"version", "profiles"})
        self.assertEqual(set(stored["profiles"]["alice"]), set(Facts.model_fields))

    def test_real_langchain_json_parser(self):
        extraction, response = demo_chains([{"name": "Анна"}], ["Здравствуйте."])
        reply = FactAgent(self.memory, extraction, response).send("Я Анна")
        self.assertEqual(reply, "Здравствуйте.")
        self.assertEqual(self.memory.load().name, "Анна")

    def test_chatopenai_structured_output_with_mock_http(self):
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "test", "object": "chat.completion", "created": 0, "model": "test",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": '{"name":"Анна","stack":["Python"]}',
                }}],
            })
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            model = ChatOpenAI(model="test", api_key="test", base_url="http://test.invalid/v1",
                               http_client=client, max_retries=0)
            chain = EXTRACTION_PROMPT | model.with_structured_output(Facts, method="json_mode")
            result = chain.invoke({"facts": "{}", "question": "Как вас зовут?", "message": "Я Анна"})
        self.assertEqual(result.name, "Анна")
        self.assertEqual(requests[0]["response_format"], {"type": "json_object"})

    def test_live_configuration_requires_model(self):
        with patch.dict(os.environ, {"LLM_MODEL": ""}):
            with self.assertRaises(ValueError):
                build_live_chains()

    def test_restart_in_separate_processes(self):
        document = run_demo()
        self.assertEqual(document["profiles"]["demo-user"]["name"], "Анна")
        self.assertIsNotNone(document["profiles"]["demo-user"]["goal"])

    def test_lab1_tools_and_prompt_reused(self):
        self.assertEqual(len(lab1.FUNCTIONS), 6)
        self.assertEqual(lab1.call_function("get_deposit_rate", "ВТБ"), "Втб: ставка по вкладу 16.50% годовых")
        for version in ("chat", "v1", "v2", "v3"):
            prompt = make_reply_prompt(version).invoke({"facts": "{}", "turn": []})
            self.assertEqual(prompt.messages[0].content, lab1.PROMPTS[version])

    def test_financial_chain_calls_two_tools_and_keeps_debug_out_of_reply(self):
        self.memory.update(Facts(name="Анна", savings_amount="1000", currency="USD", preferred_bank="ВТБ"))
        events, inputs = [], []
        replies = iter([
            "<|call|>get_exchange_rate(USD)<|call|>",
            "<tool_call>get_deposit_rate(ВТБ)</tool_call>",
            "Учебный расчёт: 92 500 рублей, ставка ВТБ 16.50%.",
        ])
        def respond(data):
            inputs.append({**data, "turn": list(data["turn"])})
            return next(replies)
        agent = FactAgent(self.memory, RunnableLambda(lambda _: Facts()), RunnableLambda(respond), events.append)
        before = self.path.read_bytes()
        reply = agent.send("Пересчитай мои сбережения и покажи ставку моего банка")
        self.assertIn("92 500", reply)
        self.assertNotIn("<|call|>", reply)
        self.assertEqual([e["name"] for e in events if e["event"] == "tool_call"],
                         ["get_exchange_rate", "get_deposit_rate"])
        self.assertIn("92.50", inputs[1]["turn"][-1].content)
        self.assertIn("16.50", inputs[2]["turn"][-1].content)
        self.assertEqual(json.loads(inputs[0]["facts"])["savings_amount"], "1000")
        self.assertEqual(self.path.read_bytes(), before)  # No tool output or chat in memory.

    def test_tool_limit_stops_repeated_calls(self):
        events = []
        agent = FactAgent(self.memory, RunnableLambda(lambda _: Facts()),
                          RunnableLambda(lambda _: "<|call|>get_time()<|call|>"), events.append)
        with self.assertRaises(RuntimeError):
            agent.send("Который час?")
        self.assertEqual(sum(e["event"] == "tool_call" for e in events), lab1.MAX_STEPS)

    def test_broken_tool_marker_is_not_shown_as_final_answer(self):
        agent = FactAgent(self.memory, RunnableLambda(lambda _: Facts()),
                          RunnableLambda(lambda _: "<|call|>broken"))
        with self.assertRaises(ValueError):
            agent.send("Курс?")

    def test_all_lab1_questions_complete_with_expected_tool_sequences(self):
        cases = [
            ([('get_exchange_rate', 'CNY')], 'Курс юаня: 12.80 RUB.'),
            ([('get_exchange_rate', 'USD')], '250 USD = 23 125 рублей.'),
            ([('get_deposit_rate', 'Сбербанк'), ('get_deposit_rate', 'Тинькофф')], '17% больше 16%.'),
            ([('get_key_rate', ''), ('get_inflation', '2024')], '16% больше 9.5%.'),
            ([('get_deposit_rate', 'ВТБ'), ('get_exchange_rate', 'USD')], 'Учебный расчёт по 16.50% и 92.50 RUB/USD.'),
            ([], 'Для биткоина инструмента нет.'),
        ]
        questions = (Path(__file__).resolve().parents[1] / '1 lab' / 'questions.md').read_text(encoding='utf-8').splitlines()
        self.assertEqual(len(questions), len(cases))
        for question, (calls, final) in zip(questions, cases):
            with self.subTest(question=question):
                events = []
                replies = [f'<|call|>{name}({arg})<|call|>' for name, arg in calls] + [final]
                extraction, response = demo_chains([{}], replies)
                agent = FactAgent(self.memory, extraction, response, events.append)
                self.assertEqual(agent.send(question), final)
                self.assertEqual([(e['name'], e['argument']) for e in events if e['event']=='tool_call'], calls)


if __name__ == "__main__":
    unittest.main()
