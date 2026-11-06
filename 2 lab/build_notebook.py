"""Generate lab 2 notebook; financial tools are imported from lab 1."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Лабораторная работа 2. Финансовый ассистент: LangChain и память фактов

**Продолжение первой лабораторной:** финансовый ассистент теперь использует
LangChain и запоминает финансовый профиль пользователя между запусками.
Инструменты, учебные данные, парсер вызовов и промпты импортируются непосредственно
из `../1 lab/finance_agent.py`. Вызовы по-прежнему текстовые, без native tools API.

Файловая память содержит профиль. Извлечение фактов и ответ — две
отдельные LangChain-цепочки. После перезапуска агент читает JSON и использует
сохранённые сумму, валюту, банк и цель, не запрашивая их повторно.

| Первая лабораторная | Продолжение во второй |
|---|---|
| Финансовые функции и константы | Те же функции из исходного модуля |
| `find_call`, `call_function` | Тот же текстовый протокол вызовов |
| Промпты `chat`, `v1`, `v2`, `v3` | Выбор через `LLM_PROMPT`, плюс контекст фактов |
| Прямой вызов OpenAI-клиента | LangChain LCEL и отдельная цепочка извлечения |
| История в оперативной памяти | JSON с фактами, загрузка при запуске |
| Вызовы и ответы в консоли | Чат и служебная CELL разделены |

**Режим по умолчанию:** воспроизводимая офлайн-демонстрация с тестовыми ответами
модели. Это проверка механики памяти, а не качества LLM. Для настоящего диалога
настройте `.env`, включите `RUN_LIVE` и выполните ячейки по порядку.

## 1. Окружение
Python 3.12. Выберите kernel виртуального окружения проекта. При необходимости
раскомментируйте установку ниже. Инструкции запуска есть в `readme.md`.
""")
code('# %pip install -r requirements.txt')
code('''
from pathlib import Path
import sys

LAB = Path.cwd()
if not (LAB / "fact_agent.py").exists():
    LAB = LAB / "2 lab"
if not (LAB / "fact_agent.py").exists():
    raise RuntimeError("Откройте notebook из папки 2 lab или корня репозитория.")
LAB = LAB.resolve()
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

RUN_LIVE = False
USER_ID = "my-profile"
MEMORY_PATH = LAB / "memory.local.json"
''')
md("""
## 2. Реализация
Ниже находится код расширения агента, а его копия в `fact_agent.py` используется
автоматическими тестами и дочерними процессами демонстрации.

Используется LCEL (`prompt | model | parser`), разрешённая заданием цепочка.
В RAM агент держит три последние пары вопрос–ответ и краткие выдержки из
четырёх предыдущих пар. В JSON сохраняются явно сообщённые факты.
""")
source = (ROOT / "fact_agent.py").read_text(encoding="utf-8")
for part in source.split("# %% "):
    if part.startswith(('Schema', 'LangChain', 'Agent')):
        title, part = part.split("\n", 1)
        md("### " + title)
    code(part)

md("""
## 3. Служебная CELL
Этот блок вывода используется для загруженных/сохранённых фактов,
вызовов финансовых инструментов и их результатов, состояния сессии и ошибок.
Он находится в отдельной ячейке от основного чата.
Исходные тексты исключений API не выводятся: они могут содержать детали запроса.
""")
code('''
import ipywidgets as widgets
from IPython.display import display

debug_output = widgets.Output(layout={"border": "1px solid #bbb", "padding": "8px"})
display(debug_output)

def show_debug(event):
    with debug_output:
        print(json.dumps(event, ensure_ascii=False, indent=2))

show_debug({"mode": "live" if RUN_LIVE else "offline demo", "user_id": USER_ID})
''')
md("""
## 4. Основной чат
Включите `RUN_LIVE = True` в первой ячейке настройки, затем выполните notebook
с начала. В `.env` укажите модель, адрес API и при необходимости ключ.
Обычное сообщение вызывает извлечение фактов и генерацию ответа.
Команда `/remember {"level":"beginner"}` записывает факт прямо в профиль.
Для завершения введите `/exit`; затем можно перезапустить kernel и снова
выполнить ячейки: `memory.local.json` будет загружен автоматически.
""")
code('''
chat_output = widgets.Output(layout={"border": "1px solid #bbb", "padding": "8px"})
message_input = widgets.Textarea(placeholder="Ваш ответ или /exit", layout={"width": "100%"})
send_button = widgets.Button(description="Отправить", button_style="primary")
agent = None
send_button.disabled = True
message_input.disabled = True

def send_message(_):
    message = message_input.value.strip()
    if not message or agent is None:
        return
    send_button.disabled = True
    message_input.disabled = True
    with chat_output:
        print("Вы:", message)
    try:
        answer = agent.send(message)
        with chat_output:
            print("Агент:", answer, "\\n")
        message_input.value = ""
    except Exception as exc:
        show_debug({"event": "request_failed", "error_type": type(exc).__name__,
                    "hint": "Проверьте API, модель и JSON-режим. Факты, сохранённые до ошибки ответа, остаются в файле."})
        with chat_output:
            print("Агент: Не удалось завершить ответ. Попробуйте ещё раз.")
    finally:
        send_button.disabled = agent.closed
        message_input.disabled = agent.closed

send_button.on_click(send_message)
display(widgets.VBox([chat_output, message_input, send_button]))
if RUN_LIVE:
    try:
        extraction, response = build_live_chains(LAB / ".env")
        agent = FactAgent(FactMemory(MEMORY_PATH, USER_ID), extraction, response, show_debug)
        with chat_output:
            print("Агент:", agent.greeting())
        send_button.disabled = False
        message_input.disabled = False
    except Exception as exc:
        show_debug({"event": "startup_failed", "error_type": type(exc).__name__,
                    "hint": "Проверьте .env, LLM_MODEL и формат файла памяти."})
else:
    with chat_output:
        print("Для диалога с моделью включите RUN_LIVE. Офлайн-проверка находится ниже.")
''')

md("""
## 5. Демонстрация настоящего перезапуска процесса
Здесь подменена только модель: `FakeListChatModel` выдаёт заранее заданные
JSON и ответы, которые проходят настоящие промпты и парсер LangChain.

1. Первый процесс сохраняет имя Анна, сбережения 1000 USD, цель и банк ВТБ, затем завершается.
2. Второй процесс читает тот же файл и отвечает на вопрос о сбережениях без повторного уточнения суммы.
3. Выполняется настоящая `get_exchange_rate(USD)` из первой лабы; учебный курс 92.50
   используется в ответе: 1000 × 92.50 = 92 500 рублей.

Демонстрация работает во временной папке и не меняет память реального пользователя.
""")
code('''
from demo_restart import run_demo
demo_document = run_demo()
assert demo_document["profiles"]["demo-user"]["name"] == "Анна"
assert demo_document["profiles"]["demo-user"]["goal"] is not None
''')
md("""
### Окно, сводка и выбор сложности
В этом примере окно уменьшено до одной пары. Третий запрос получает выдержку
из первого обмена и полный второй обмен. Ответы модели заданы для демонстрации.
""")
code('''
from langchain_core.runnables import RunnableLambda

def inspect_request(data):
    print("Стиль:", data["teaching_style"])
    print("Контекст:", [msg.content for msg in data["context"]])
    return next(answers)

with tempfile.TemporaryDirectory() as folder:
    answers = iter(["Вклад — деньги в банке под проценты. Срок задаёт договор.",
                    "Ставка задаёт процент за год.", "Доход зависит от срока вклада."])
    sample = FactAgent(FactMemory(Path(folder) / "profile.json", "demo"),
                       RunnableLambda(lambda _: {}), RunnableLambda(inspect_request), window=1)
    print(sample.send('/remember {"level":"beginner"}'))
    print(sample.send("Что такое вклад?"))
    print(sample.send("Что означает ставка?"))
    print(sample.send('/remember {"level":"advanced"}'))
    print(sample.send("Как связан доход со сроком?"))
    assert sample.memory.load().level == "advanced"
''')
md("""
## 6. Файл памяти для сдачи
`memory.json` — пример с вымышленным профилем из демонстрации.
Рабочая память `memory.local.json` создаётся при первом сохранении факта в живом
чате и исключена из Git. Для нескольких пользователей измените `USER_ID`.
""")
code('''
example_document = json.loads((LAB / "memory.json").read_text(encoding="utf-8"))
assert example_document == demo_document
print(json.dumps(example_document, ensure_ascii=False, indent=2))
''')
md("""
## 7. Проверка с настоящей моделью
Для проверки качества извлечения необходимо отдельно запустить API модели:

1. Включить `RUN_LIVE`, начать с новым `USER_ID`.
2. Написать: «Меня зовут Анна, у меня 1000 долларов, цель — финансовая подушка, мой банк ВТБ».
3. Убедиться, что в служебной CELL появились имя, сумма, валюта, цель и банк.
4. Отправить `/exit`, выполнить **Restart Kernel**, затем **Run All**.
5. Спросить: «Сколько моих сбережений получится в рублях?» — сумма и валюта уже известны.
6. Проверить вызов `get_exchange_rate(USD)` в служебной CELL и итоговый ответ в чате.
7. Написать: «Теперь у меня 1500 долларов»; перезапустить kernel и проверить новую сумму.
8. Повторить вопросы из `../1 lab/questions.md`: инструменты должны продолжать работать.

## Вывод и ограничения
Файловая память и продолжение опроса проверяются в двух отдельных процессах.
Извлечение и ответы настоящей LLM зависят от выбранной модели; офлайн-проверка
не подтверждает их качество. Совместимость клиента с JSON-ответом API проверяется
тестом с подменённым HTTP-транспортом. Автотесты: `python -m unittest -v`.

Файл хранит профиль: имя, цель, сбережения, валюту, банк, страну, срок,
а также сообщённые уровень и стек. Переписка и результаты функций в файл не пишутся.
Последний ответ используется для коротких уточнений только внутри текущей сессии.
Одновременно файл должен изменять
один процесс; для многопользовательского сервера нужна БД или блокировка записи.
Удаление отдельных фактов командами чата не реализовано. При сбое извлечения
память не меняется; при сбое генерации уже извлечённые факты сохраняются.

Документация: [LCEL Runnable](https://reference.langchain.com/python/langchain-core/runnables/base),
[ChatOpenAI structured output](https://reference.langchain.com/python/langchain-openai/chat_models/base/ChatOpenAI/with_structured_output).
""")
notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {"display_name": "Python 3 (MLOps labs)", "language": "python", "name": "python3"}
notebook.metadata.language_info = {"name": "python", "version": "3.12"}
nbf.validate(notebook)
nbf.write(notebook, ROOT / "lab2_fact_memory.ipynb")
print("Created lab2_fact_memory.ipynb")
