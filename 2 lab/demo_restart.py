"""Deterministic offline check; model replies are test fixtures, not real inference."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

from fact_agent import (
    EXTRACTION_PROMPT, REPLY_PROMPT, FactAgent, FactMemory, Facts,
)


def demo_chains(patches, replies):
    extraction = (
        EXTRACTION_PROMPT
        | FakeListChatModel(responses=[json.dumps(patch, ensure_ascii=False) for patch in patches])
        | PydanticOutputParser(pydantic_object=Facts)
    )
    response = REPLY_PROMPT | FakeListChatModel(responses=replies) | StrOutputParser()
    return extraction, response


def worker(stage, path):
    memory = FactMemory(path, "demo-user")
    if stage == "write":
        extraction, response = demo_chains(
            [{"name": "Анна", "savings_amount": "1000", "currency": "USD",
              "goal": "Накопить финансовую подушку", "preferred_bank": "ВТБ"}],
            ["Анна, запомнила ваши сбережения 1000 USD, цель и банк ВТБ."],
        )
        agent = FactAgent(memory, extraction, response)
        print("Агент:", agent.greeting())
        message = "Меня зовут Анна. У меня 1000 долларов, хочу накопить финансовую подушку. Предпочитаю ВТБ."
        print("Пользователь:", message)
        print("Агент:", agent.send(message))
        print("Пользователь: /exit")
        print("Агент:", agent.send("/exit"))
    else:
        assert memory.load().name == "Анна"
        assert memory.load().savings_amount == "1000"
        assert memory.load().currency == "USD"
        events = []
        extraction, response = demo_chains(
            [{}],
            ["<|call|>get_exchange_rate(USD)<|call|>",
             "Анна, ваши 1000 USD — это 92 500 рублей по учебному курсу 92.50 RUB/USD из первой лабы."],
        )
        agent = FactAgent(memory, extraction, response, events.append)
        greeting = agent.greeting()
        assert "Анна" in greeting and "Как вас зовут?" not in greeting
        print("Агент после перезапуска:", greeting)
        message = "Сколько моих сбережений получится в рублях?"
        print("Пользователь:", message)
        print("Агент:", agent.send(message))
        calls = [event for event in events if event["event"] == "tool_call"]
        assert len(calls) == 1 and calls[0]["name"] == "get_exchange_rate"
        assert calls[0]["argument"] == "USD" and "92.50" in calls[0]["result"]
        print("Служебная проверка: вызвана настоящая get_exchange_rate(USD) из 1 lab.")
        print("PASS: сумма и валюта восстановлены из файла, инструмент первой лабы выполнен.")


def run_demo():
    print("OFFLINE DEMO: ответы модели заданы заранее; проверяется цепочка и файловая память.")
    # Temporary memory keeps the submitted example and real user's profile intact.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "memory.json"
        for stage in ("write", "read"):
            print(f"\nОтдельный процесс: {stage}")
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--stage", stage, "--memory", str(path)],
                check=True, capture_output=True, text=True, encoding="utf-8",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            print(result.stdout.strip())
        print("\nИтоговый файл памяти (без истории сообщений):")
        print(path.read_text(encoding="utf-8"))
        return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["write", "read"])
    parser.add_argument("--memory", type=Path)
    args = parser.parse_args()
    if args.stage:
        if args.memory is None:
            parser.error("--stage requires --memory")
        worker(args.stage, args.memory)
    else:
        run_demo()
