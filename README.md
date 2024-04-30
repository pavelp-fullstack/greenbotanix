# GreenBotanix: 

## Cтруктура проекта

```
├───api (сервис REST API)
│   └─── // исходный код и конфигурация сервиса API
│
├───chatbot (сервис Telegram bot)
│   └─── // исходный код и конфигурация сервиса chatbot
│
├───data (рабочие данные проекта)
│   ├──── db //сюда мамятся данные для работы API
│   ├──── imported_docs //сюда API сохраняет загруженные документы в исходном формате
│   └──── training_docs //отсюда скрипт upload_docs.bat загружает в API документы поддерживаемых типов
│
├───explore (ноутбуки jupyter и другие файлы для экспериметов и подготовки датасетов)
│
├───utils (вспомогательные скрипты)
│
скрипты и конфиги общие для всего проекта

```

Код структурирован так чтобы не было зависимостей по коду между сервисами и вообще другими фолдерами проекта. Параметры настройки вынесены в .env файлы (которые надо создать перед запуском сервисов проекта - см файлы template.env в качестве шаблонов).
Корневой .env задает параметры для сервисов при выполнении в Docker контейнерах.
Файлы внутри проектов ./api/.env, ./chatbot/.env необязательны но их удобно использовать в среде разработки. Для удобства отладки сервисов можно сконфигурировать загрузку переменных окружения из этих файлов (<project>/.env) перед запуском проекта - в этом случае приоритет будут иметь переменные из этих файлов.

Если в результате разработки поменялся перечень используемых пакетов то изменения следует зафиксировать выполнив команду
pip freeze >requirements.txt в целевом фолдере сервиса. При этом виртуальное окружение должно быть активировано.

Для тестирования отдельных сервисов их контейнеры можно собирать и запускать по отдельности - соотв скриптами *.bat из корня проекта. Пересборку контейнеров надо запускать в том случае если изменился код и/или список зависимостей проекта

Следует поддерживать чистоту папок сервисов  сохраняя там только рабочий код. Файлы данных должны помещаться в фолдеры, вложенные в ./data. Скрипты (или ноутбуки) для экспериментов должны сохраняться в ./explore

Документацию следует помещать в ./docs

# Локальная конфигурация среды разработки

Следующая последовательность выполняется единожды. Уже выполненные шаги следует пропустить.

01. Установить и сконфиурировать Git
02. Установить и сконфигурировать Docker
03. Установить и сконфигурировать conda (Anaconda или Miniconda)
04. Создать окружение в conda для python 3.10 командой - conda create -n P3.10 python=3.10
05. Сделать его текущим - conda activate P3.10
06. Склонировать проект из репо командой git clone (*использовать урл по протоколу ssh - git://....*)
07. Перейти в целевой подфолдер с которым подразумевается работа в контексте текущей задачи - explore, api или chatbot
09. Создать виртуальное окружение в этом подфолдере командой python -m venv .venv - если еще не создано (папка .venv отсутствует)
10. Активировать окружение командой  *.\.venv\scripts\activate* - при успешном выполнение должен поменяться формат промпта на (.venv)\....
11. Установить пакеты командой python -m pip install -r requirements.txt

# Как работать c проектом

1. Рабочий код содержится в ветке main - в нее должны попадать только рабочие изменения и только в минимально необходимом количестве
2. Для изменения репозитория в рамках работы над конкретной задачей следует придерживаться следующей последовательности действий:
    2.1 Обновить локальную ветку main командой git pull - если работа ведется верно то конфликтов на этой стадии не должно быть
    2.2 Создать ветку для задачи от ветки main командой - git checkout -b <prefix>/<digest>, где <prefix> одно из:
        - docs - если изменение касается документации
        - feature - если добавляется функционал
        - bugfix - если изменение касается исправлений ошибок
        - test - если добавляются тесты или утилиты - главный код проекта не меняется

        <digest> - краткое описание что меняется - без пробелов по англ до 50 символов

3. После успешного выполнения команды (проверить вывод в консоль) 
    - перейти в целевой фолдер, 
    - активировать вирт окружение, 
    - если надо обновить зависимости (при изменении файла requirements.txt) командой python -m pip install -r requirements.txt
    - внести требуемые изменения

4. Если список зависимостей был изменен - добавлены или удалены пакеты то зафиксировать рабочий список зависимостей командой - python -m pip freeze >requirements.txt

5. Добавить и закомитить изменения *после проверки работоспобосности внесенных изменений*
6. Обновить ветку main еще раз командой git pull
7. Вернуться в свою текущую ветку и выполнить команду git rebase --onto main для того чтобы применить последние изменения из этой ветки (если они есть). Если возникли конфликты то разрезолвить их и завершить операцию rebase
8. Запушить локальные изменения в репо командой git push
10. Создать pull request из своей ветки в main - web консоль github или клиент github 
11. Попросить коллег сделать ревью 



# Конфигурация API


Tестовая консоль (OpenAPI/Swagger) доступна по URI - /api/explore - в ней можно протестировать основные эндпойнты
По умолчанию полный URL - http://localhost:8000/api/explore

Операции для работы с API:
```
TBD
```


# Конфигурация Telegram бота 

- Найдите бота `@BotFather` в Telegram и начните диалог. 
- Создайте нового бота, следуя инструкциям  `@BotFather` . 
- Получите уникальный API токен для вашего бота. 
- Установите значение `GB_CHATBOT_TOKEN=<real telegram api token here>` в файле `.env` в корне проекта (по общей структуре файла см прототип template.env), указав полученный API токен. 

При подсоединении к боту - если приветствие не показалось ввести сообщение /start
