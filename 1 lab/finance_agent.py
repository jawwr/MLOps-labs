"""Финансовый ассистент с вызовом функций из текста ответа.

Работает с локальным llama-server (llama-cpp) и с OpenRouter через
OpenAI-совместимый API. Нативный function calling / tools / MCP не используется.

Запуск:
    python finance_agent.py --prompt v2
    LLM_BASE_URL=https://openrouter.ai/api/v1 LLM_API_KEY=sk-or-... \\
        LLM_MODEL=openai/gpt-4o-mini python finance_agent.py --prompt v3
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI, OpenAIError

log = logging.getLogger(__name__)

load_dotenv()

# ---------------------------------------------------------------------------
# Данные (захардкоженные константы — как требует задание 4)
# ---------------------------------------------------------------------------

EXCHANGE_RATES: dict[str, float] = {
    "USD": 92.50,
    "EUR": 100.10,
    "CNY": 12.80,
    "KZT": 0.19,
}

DEPOSIT_RATES: dict[str, float] = {
    "сбербанк": 16.0,
    "втб": 16.5,
    "тинькофф": 17.0,
    "альфа": 16.8,
}

KEY_RATE = 16.0  # % годовых
INFLATION: dict[int, float] = {2022: 11.9, 2023: 7.4, 2024: 9.5, 2025: 8.0}
TAX_RATES: dict[str, float] = {"RU": 13.0, "US": 22.0, "DE": 45.0}


# ---------------------------------------------------------------------------
# Инструменты
# ---------------------------------------------------------------------------


def get_time(*_args) -> str:
    """Текущее время — удобно для отладки агента."""
    return datetime.datetime.now().strftime("%H:%M:%S")


def get_exchange_rate(currency: str) -> str:
    code = currency.strip().upper()
    rate = EXCHANGE_RATES.get(code)
    if rate is None:
        return f"нет данных о курсе {code or '???'}"
    return f"1 {code} = {rate:.2f} RUB"


def get_deposit_rate(bank: str) -> str:
    name = bank.strip().lower()
    rate = DEPOSIT_RATES.get(name)
    if rate is None:
        return f"нет данных о вкладах в «{bank}»"
    return f"{bank.title()}: ставка по вкладу {rate:.2f}% годовых"


def get_key_rate(*_args) -> str:
    return f"Ключевая ставка ЦБ РФ: {KEY_RATE:.2f}% годовых"


def get_inflation(year: str) -> str:
    try:
        y = int(year.strip())
    except ValueError:
        return f"«{year}» — не похоже на год"
    rate = INFLATION.get(y)
    if rate is None:
        return f"нет данных об инфляции за {y} год"
    return f"Инфляция в {y} году: {rate:.1f}%"


def get_tax_rate(country: str) -> str:
    code = country.strip().upper()
    rate = TAX_RATES.get(code)
    if rate is None:
        return f"нет данных о налогах в «{country}»"
    return f"Ставка налога в {code}: {rate:.1f}%"


FUNCTIONS = {
    "get_time": get_time,
    "get_exchange_rate": get_exchange_rate,
    "get_deposit_rate": get_deposit_rate,
    "get_key_rate": get_key_rate,
    "get_inflation": get_inflation,
    "get_tax_rate": get_tax_rate,
}


# ---------------------------------------------------------------------------
# Парсинг вызова
# ---------------------------------------------------------------------------

MARKER = "<|call|>"
CALL_PATTERNS = [
    re.compile(rf"{re.escape(MARKER)}\s*(\w+)\s*\(([^)]*)\)\s*{re.escape(MARKER)}"),
    re.compile(r"<\|?tool_call\|?>\s*(\w+)\s*\(([^)]*)\)\s*<\|?/?tool_call\|?>", re.I),
    re.compile(r"<\|?call\|?>\s*(\w+)\s*\(([^)]*)\)\s*<\|?/?call\|?>", re.I),
    re.compile(r"\bfunction\s+(\w+)\s*\(([^)]*)\)", re.I),
]

def find_call(reply: str) -> tuple[str, str] | None:
    for pattern in CALL_PATTERNS:
        m = pattern.search(reply)
        if m:
            name, raw = m.groups()
            argument = raw.rpartition("=")[2].strip().strip("\"'")
            return name, argument
    return None


def call_function(name: str, argument: str) -> str:
    function = FUNCTIONS.get(name)
    if function is None:
        return f"error: unknown function {name!r}; available: {', '.join(FUNCTIONS)}"
    try:
        result = function(argument)
    except Exception as exc:
        return f"error: {exc}"
    if isinstance(result, (dict, list)):
        return json.dumps(result, ensure_ascii=False)
    return str(result)


# ---------------------------------------------------------------------------
# Промпты — это ядро задания 4.2
# ---------------------------------------------------------------------------

PROMPT_CHAT = """\
Ты — финансовый консультант. Отвечай кратко, по делу, на русском языке."""

PROMPT_V1 = """\
Ты — финансовый ассистент с доступом к функциям.

Чтобы вызвать функцию, напиши ровно:
<|call|>имя(аргумент)<|call|>

Доступные функции:
- get_time() — текущее время
- get_exchange_rate(currency) — курс валюты к рублю (USD, EUR, CNY, KZT)
- get_deposit_rate(bank) — ставка по вкладу (Сбербанк, ВТБ, Тинькофф, Альфа)
- get_key_rate() — ключевая ставка ЦБ
- get_inflation(year) — инфляция за год
- get_tax_rate(country) — налоговая ставка (RU, US, DE)

Всегда отвечай на русском языке."""

PROMPT_V2 = """\
Ты — финансовый ассистент с доступом к функциям. Когда пользователю нужны числа \
(курс, ставка, инфляция, налог), которых у тебя нет, ты вызываешь функцию.

<format>
Вызов функции — это отдельное сообщение ровно такого вида:
<|call|>get_exchange_rate(USD)<|call|>
Результат придёт следующим сообщением, начиная со слова «Результат».
Не пиши результат сам, не выдумывай числа.
</format>

<functions>
- get_time() — текущее время
- get_exchange_rate(currency) — курс валюты к рублю (USD, EUR, CNY, KZT)
- get_deposit_rate(bank) — ставка по вкладу (Сбербанк, ВТБ, Тинькофф, Альфа)
- get_key_rate() — ключевая ставка ЦБ
- get_inflation(year) — инфляция за год
- get_tax_rate(country) — налоговая ставка (RU, US, DE)
</functions>

<rules>
- Если нужно сравнить или посчитать — сначала собери все данные через вызовы, потом считай.
- Арифметику делай сам, но только над числами, полученными из функций.
- Всегда отвечай на русском языке.
</rules>

<example>
Пользователь: Какой сейчас курс доллара?
Ассистент: <|call|>get_exchange_rate(USD)<|call|>
Пользователь: Результат get_exchange_rate(USD): 1 USD = 92.50 RUB
Ассистент: Курс доллара: 92.50 ₽.
</example>"""

PROMPT_V3 = """\
Ты — финансовый ассистент с доступом к функциям. Ты НИКОГДА не выдумываешь числа — \
только берёшь их из результата функции.

<algorithm>
На каждый вопрос пользователя:
1. Разбей вопрос на факты, которые нужно узнать (курс, ставка, инфляция, налог).
2. Для каждого факта ответь ТОЛЬКО вызовом функции:
   <|call|>имя(аргумент)<|call|>
3. Дождись результата. Результат приходит в сообщении «Результат имя(аргумент): ...».
4. Повторяй шаг 2, пока все нужные числа не собраны.
5. Когда данных достаточно — посчитай и дай финальный ответ с числами и единицами.
</algorithm>

<functions>
- get_time() — текущее время. Без аргументов.
- get_exchange_rate(currency) — курс валюты к рублю. Аргумент — код валюты (USD, EUR, CNY, KZT).
- get_deposit_rate(bank) — ставка по вкладу. Аргумент — название банка (Сбербанк, ВТБ, Тинькофф, Альфа).
- get_key_rate() — ключевая ставка ЦБ. Без аргументов.
- get_inflation(year) — инфляция за год. Аргумент — год (2022, 2023, 2024, 2025).
- get_tax_rate(country) — налоговая ставка. Аргумент — код страны (RU, US, DE).
</functions>

<rules>
- Вызов функции — это отдельное сообщение, ничего кроме вызова.
- Не пиши результат функции от себя — он придёт в сообщении «Результат ...».
- Если для ответа нужно два числа — сделай два вызова подряд, не пытайся угадать.
- Всегда указывай единицы: рубли, проценты, годы.
- Всегда отвечай на русском языке.
</rules>"""

PROMPTS = {"chat": PROMPT_CHAT, "v1": PROMPT_V1, "v2": PROMPT_V2, "v3": PROMPT_V3}


# ---------------------------------------------------------------------------
# Ассистент
# ---------------------------------------------------------------------------

MAX_STEPS = 6           # финансовые вопросы часто требуют 3–4 вызовов подряд
EXIT_COMMAND = "/exit"
LOG_DIR = Path(__file__).parent / "logs"


class Assistant:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        system_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

    def ask(self, question: str) -> None:
        log.info("user: %s", question)
        turn = [{"role": "user", "content": question}]
        for _ in range(MAX_STEPS):
            reply = self._complete(self.history + turn)
            call = find_call(reply)
            if call is None:
                if MARKER in reply:
                    log.warning("в ответе есть маркер, но вызов не распознан: %s", reply)
                self.history += [*turn, {"role": "assistant", "content": reply}]
                return
            name, argument = call
            result = call_function(name, argument)
            log.info("call %s(%s) -> %s", name, argument, result)
            print(f"-> {result}")
            turn += [
                {"role": "assistant", "content": f"{MARKER}{name}({argument}){MARKER}"},
                {"role": "user", "content": f"Результат {name}({argument}): {result}"},
            ]
        log.warning("нет ответа после %d вызовов функций", MAX_STEPS)
        print(f"(нет ответа после {MAX_STEPS} вызовов функций)")

    def _complete(self, messages: list[dict[str, str]]) -> str:
        started = time.perf_counter()
        parts: list[str] = []
        finish_reason = None
        with self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        ) as stream:
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.delta.content:
                    print(choice.delta.content, end="", flush=True)
                    parts.append(choice.delta.content)
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
        print()
        reply = "".join(parts)
        log.info("assistant (%.1fs): %s", time.perf_counter() - started, reply)
        if finish_reason == "length":
            log.warning("ответ обрезан по max_tokens")
        return reply


# ---------------------------------------------------------------------------
# Подключение, логирование, REPL
# ---------------------------------------------------------------------------


def connect() -> tuple[OpenAI, str]:
    base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8081/v1")
    api_key = os.getenv("LLM_API_KEY", "none")
    headers: dict[str, str] = {}
    if "openrouter" in base_url:
        headers = {
            "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_TITLE", "Finance Assistant"),
        }
    client = OpenAI(base_url=base_url, api_key=api_key, default_headers=headers)
    model = os.getenv("LLM_MODEL", "llama-3.2-3b")
    if not model:
        try:
            model = client.models.list().data[0].id
        except OpenAIError as exc:
            sys.exit(f"error: cannot query /v1/models ({exc}); задайте LLM_MODEL")
    return client, model


def setup_logging(model: str, prompt: str) -> None:
    name = re.sub(r"[^\w.-]", "-", Path(model).name.removesuffix(".gguf"))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_DIR / f"{name}.{prompt}.log",
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def repl(assistant: Assistant) -> None:
    print(f"Модель: {assistant.model}. {EXIT_COMMAND} или Ctrl+D — выход.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question == EXIT_COMMAND:
            return
        if not question:
            continue
        try:
            assistant.ask(question)
        except KeyboardInterrupt:
            log.warning("ответ прерван")
            print("\n(прервано)")
        except OpenAIError as exc:
            log.error("request failed: %s", exc)
            print(f"\nerror: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Финансовый ассистент с вызовом функций из текста ответа.",
        epilog=(
            "Модель задаётся LLM_BASE_URL, LLM_API_KEY, LLM_MODEL. "
            f"{EXIT_COMMAND} или Ctrl+D — выход."
        ),
    )
    parser.add_argument("--prompt", choices=PROMPTS, default="v2")
    parser.add_argument("-t", "--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    try:
        client, model = connect()
    except OpenAIError as exc:
        sys.exit(f"error: {exc}")

    setup_logging(model, args.prompt)
    log.info(
        "model=%s prompt=%s temperature=%s max_tokens=%s",
        model, args.prompt, args.temperature, args.max_tokens,
    )
    print(f"model: {model}")
    repl(Assistant(client, model, PROMPTS[args.prompt], args.temperature, args.max_tokens))


if __name__ == "__main__":
    main()