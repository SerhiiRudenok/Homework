import os
import requests
import speech_recognition as sr # Розпізнавання голосу
from gtts import gTTS # Перетворення тексту в голос = озвучка
import tempfile
import platform
import subprocess
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
import pygame
from langdetect import detect
from textblob import TextBlob
import random


# --- Конфігурація ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or "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"
if not GROQ_API_KEY:
    raise RuntimeError("Відсутній API ключ. Будь ласка, встановіть змінну середовища GROQ_API_KEY.")

MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Системні промпти для різних мов
system_prompts = {
    "uk": "Ти - розумний і доброзичливий у спілкуванні український асистент. Відповідай коротко і просто.",
    "en": "You are a smart and friendly Ukrainian assistant. Answer briefly and simply.",
    "pl": "Jesteś inteligentnym i sympatycznym ukraińskim asystentem. Odpowiadaj krótko i zwięźle.",
    "fr": "Vous êtes un assistant ukrainien intelligent et sympathique. Répondez brièvement et simplement."
}

# Словник мови
speech_lang_codes = {
    "en": "en-US",
    "pl": "pl-PL",
    "fr": "fr-FR",
    "uk": "uk-UA"
}

# Локалізовані повідомлення
localized_messages = {
    "uk": {
        "speak": "Говори щось українською...",
        "timeout": "Час очікування вичерпано, нічого не почув.",
        "no_input": "Нічого не почув!",
        "you_said": "Ти сказав: {}",
        "think": "Думаю...",
        "goodbye": "До побачення!",
    },
    "en": {
        "speak": "Say something in English...",
        "timeout": "Timeout reached, nothing was heard.",
        "no_input": "Didn't hear anything!",
        "you_said": "You said: {}",
        "think": "Thinking...",
        "goodbye": "Goodbye!",
    },
    "pl": {
        "speak": "Powiedz coś po polsku...",
        "timeout": "Przekroczono czas oczekiwania, nic nie usłyszano.",
        "no_input": "Nic nie usłyszano!",
        "you_said": "Powiedziałeś: {}",
        "think": "Myślę...",
        "goodbye": "Do widzenia!",
    },
    "fr": {
        "speak": "Parlez en français...",
        "timeout": "Temps écoulé, rien entendu.",
        "no_input": "Je n'ai rien entendu !",
        "you_said": "Vous avez dit : {}",
        "think": "Je réfléchis...",
        "goodbye": "Au revoir!",
    }
}

# Команди для перемикання мови
switch_language = {
    "en": ["перемкни на англійську", "говори англійською", "спілкуйся англійською",  "переключи на англійську"],
    "pl": ["перемкни на польську", "говори польською", "спілкуйся польською", "переключи на польську"],
    "fr": ["перемкни на французьку", "говори французькою", "спілкуйся французькою", "переключи на французьку"],
    "uk": ["перемкни на українську", "говори українською", "спілкуйся українською", "переключи на українську", "переключись на українську"],
}

# Команди для запуску міні-гри "Вгадай число"
play_commands = [
    "міні гра", "міні-гра", "гра вгадай число", "давай пограємо", "мини гра", "мини-гра", "вгадай число"
]

# Команди для виходу
exit_commands = [
    "вихід", "вийти", "стоп", "зупинись", "припини", "завершити",
    "quit", "exit", "stop", "terminate", "bye", "goodbye",
    "do widzenia", "au revoir"
]

# Ключові слова для валют
currency_keywords = {
    "usd": ["долар", "долара", "доларів"],
    "eur": ["євро", "євра"]
}
# Функція для визначення валюти
def detect_currency(text):
    lowered = text.lower()
    for code, keywords in currency_keywords.items():
        if any(word in lowered for word in keywords):
            return code.upper()
    return None
# Функція для отримання курсу долара
def get_currency_rate(currency="USD"):
    url = "https://api.privatbank.ua/p24api/pubinfo?exchange&coursid=5"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            for item in data:
                if item["ccy"] == currency:
                    buy = item["buy"]
                    sale = item["sale"]
                    return f"Курс {currency}: купівля {buy} грн, продаж {sale} грн"
            return f"Не знайдено інформації про курс {currency}."
        else:
            return "Не вдалося отримати курс валют."
    except Exception as e:
        return f"Помилка при отриманні курсу валют: {e}"

# Функція для отримання погоди з Open-Meteo
def get_weather(city="Києві"):
    # Координати для різних міст
    city_coords = {
        "Києві": (50.45, 30.52),
        "Вінниці": (49.23, 28.48),
        "Луцьку": (50.75, 25.33),
        "Дніпрі": (48.45, 34.98),
        "Донецьку": (48.00, 37.80),
        "Житомирі": (50.25, 28.67),
        "Запоріжжі": (47.84, 35.14),
        "Івано-Франківську": (48.92, 24.71),
        "Кропивницькому": (48.51, 32.26),
        "Луганську": (48.57, 39.32),
        "Львові": (49.84, 24.03),
        "Миколаєві": (46.97, 32.00),
        "Одесі": (46.48, 30.73),
        "Полтаві": (49.59, 34.55),
        "Рівному": (50.62, 26.25),
        "Сумах": (50.91, 34.81),
        "Тернополі": (49.55, 25.59),
        "Ужгороді": (48.62, 22.30),
        "Харкові": (49.99, 36.23),
        "Херсоні": (46.63, 32.60),
        "Хмельницькому": (49.42, 27.00),
        "Черкасах": (49.44, 32.06),
        "Чернівцях": (48.29, 25.94),
        "Чернігові": (51.50, 31.30),
    }

    lat, lon = city_coords.get(city, (50.45, 30.52))  # fallback: Kyiv
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

    try:
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            weather = data["current_weather"]
            temp = weather["temperature"]
            wind = weather["windspeed"]
            return f"У місті {city}: {temp}°C, вітер {wind} км/год"
        else:
            return "Не вдалося отримати погоду з Open-Meteo."
    except Exception as e:
        return f"Помилка при запиті до Open-Meteo: {e}"

# Міні-гра "Вгадай число"
def play_number_game(current_lang = "uk"):
    target = random.randint(1, 100)
    attempts = 0

    speak_text = "Гра почалась. Спробуй вгадати число від 1 до 100!"
    print(speak_text)
    speak_multi_language(speak_text, lang=current_lang)

    while True:
        query, _ = listen_language(current_lang)
        if not query:
            continue

        # Спроба витягти число з голосового запиту
        digits = [int(word) for word in query.split() if word.isdigit()]
        if not digits:
            speak_multi_language("Будь ласка, назви число.", lang=current_lang)
            continue

        guess = digits[0]
        attempts += 1

        if guess < target:
            speak_text = "Загадане число більше."
            print(speak_text)
            speak_multi_language(speak_text, lang=current_lang)
        elif guess > target:
            speak_text = "Загадане число менше."
            print(speak_text)
            speak_multi_language(speak_text, lang=current_lang)
        else:
            speak_text = f"Вітаю! Ти вгадав число {target} за {attempts} спроб!"
            print(speak_text)
            speak_multi_language(speak_text, lang=current_lang)
            break

# Тон користувача
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.3:
        return "позитивний"
    elif polarity < -0.3:
        return "негативний"
    return "нейтральний"

# Функція для визначення команди перемикання мови
def detect_switch_command(text):
    lowered = text.lower().strip()
    for lang, phrases in switch_language.items():
        if any(phrase in lowered for phrase in phrases):
            return lang
    return None

# Розпізнавання голосу в кількох мовах
def recognize_multilang(audio):
    results = {}
    for lang_code in speech_lang_codes.values():
        try:
            text = sr.Recognizer().recognize_google(audio, language=lang_code)
            results[lang_code] = text
        except:
            continue
    return results

# Визначення мови тексту
def detect_language(text):
    try:
        lang = detect(text)
        return lang if lang in speech_lang_codes else "None"
    except:
        return "None"

# Відтворення аудіо
def play_audio(path):
    system = platform.system()
    try:
        if system == "Windows":
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)  # Чекаємо завершення
                pygame.mixer.music.stop()  # Зупинити
                pygame.mixer.quit()  # Вивільнити ресурси
                os.remove(path)  # Тепер безпечно видалити
            except Exception as e:
                print(f"Не вдалося відтворити аудіо: {e}")
        elif system == "Darwin":  # macOS
            subprocess.run(["afplay", path])
            os.remove(path)  # Видаляємо тимчасовий файл після відтворення
        else:  # Linux та інші
            subprocess.run(["mpg123", path])
            os.remove(path)  # Видаляємо тимчасовий файл після відтворення
    except Exception as e:
        print(f"Не вдалося відтворити аудіо: {e}")

# Прослуховування та розпізнавання мови
def listen_language(current_lang, timeout=7, phrase_time_limit=20):
    r = sr.Recognizer() # розпізнавач голосу
    messages = localized_messages.get(current_lang) # локалізовані повідомлення

    with sr.Microphone(device_index=1) as source:
        print(messages["speak"]) # "Говори щось українською..."
        # print(sr.Microphone.list_microphone_names()) # список мікрофонів
        r.adjust_for_ambient_noise(source, duration=0.8) # налаштування на шум
        try:
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit) # слухати
        except sr.WaitTimeoutError:
            print(messages["timeout"]) # "Час очікування вичерпано, нічого не почув."
            return None, current_lang
        except Exception:
            print(messages["no_input"]) # Нічого не почув!
            return None, current_lang
    try:
        # багатомовне розпізнавання
        candidates = recognize_multilang(audio)
        for lang_code, text in candidates.items():
            lang = detect_switch_command(text)
            if lang:
                print(f"Ти сказав: {text}")
                print(f"Виявлено команду перемикання на мову: {lang}")
                return text, lang  # повертаю текст і мову, на яку треба перемкнутись

        # Якщо не було команди перемикання, перевожу аудіо в текст поточною мовою
        text = r.recognize_google(audio, language=current_lang)  # розпізнати
        messages = localized_messages.get(current_lang, localized_messages)
        print([current_lang], messages["you_said"].format(text))
        return text, current_lang

    except Exception:
        print("Не вдалося розпізнати мову.")
    return None, current_lang

# Запит до GROQ з урахуванням мови та настрою
def ask_groq(prompt = "test", lang = "uk", sentiment="нейтральний"):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    system = system_prompts.get(lang, system_prompts["uk"]) # промпт

    # Інструкції щодо настрою
    mood_instruction = {
        "позитивний": "Користувач у гарному настрої — відповідай з ентузіазмом.",
        "негативний": "Користувач засмучений — відповідай м’яко, підтримуючи.",
        "нейтральний": "Відповідай звичайним тоном."
    }.get(sentiment, "")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": f"{system} {mood_instruction}"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7, # креативність відповіді (0-2)
        "max_tokens": 512, # макс. довжина відповіді
        "stream": False # чи потрібен стрімінг. Якщо так, то відповідь буде частинами
    }

    try:
        res = requests.post(GROQ_URL, headers=headers, json=payload)
        if res.status_code != 200:
            print("Сталась помилка!")
            print(f"Помилка: {res.status_code}, {res.text}")
            return "Вибач, сталася помилка при зверненні до Штучного Інтелекту!"
        data = res.json()
        return data["choices"][0]["message"]["content"]
    except Exception as err:
        print("Помилка при запиті:", err)
        return "Вибач, сталася помилка при зверненні до Штучного Інтелекту!"

def speak_multi_language(text, lang):
    if not text:
        return
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_path = fp.name
        tts.save(temp_path)
        play_audio(temp_path)
    except Exception as e:
        print(f"Не вдалося озвучити текст: {e}")

def main():
    print("-"*70)
    print("Привіт! Я твій голосовий асистент. Я знаю українську, англійську, польську та французьку мову. Скажи щось...")

    current_lang = "uk"

    while True:
        query, detected_lang = listen_language(current_lang)
        if not query:
            continue
        current_lang = detected_lang
        messages = localized_messages.get(current_lang)  # локалізовані повідомлення
        print(messages["think"])  # "Думаю..."

        # Перевірка на команду перемикання мови
        lowered = query.lower().strip()
        if lowered in exit_commands:
            messages = localized_messages.get(current_lang)  # локалізовані повідомлення
            speak_multi_language(messages["goodbye"], lang=current_lang) # "До побачення!"
            break
        if "погода" in lowered:
            # Визначення міста з запиту
            words = lowered.split()
            city = "Києві"  # місто за замовчуванням
            for i, word in enumerate(words):
                if (word == "у" or word == "в") and i + 1 < len(words):
                    city = words[i + 1].capitalize()
                    break
            weather_info = get_weather(city)
            print(weather_info)
            speak_multi_language(weather_info, lang=current_lang)
            continue
        # Обробка запиту про курс валюти
        currency_code = detect_currency(lowered)
        if currency_code:
            rate_info = get_currency_rate(currency_code)
            print(rate_info)
            speak_multi_language(rate_info, lang=current_lang)
            continue
        if any(cmd in lowered for cmd in play_commands):
            play_number_game()
            continue

        # Аналіз настрою та тону користувача
        sentiment = analyze_sentiment(query)
        print(f"Настрій користувача: {sentiment}")
        # Запит з урахуванням мови та настрою
        answer = ask_groq(query, lang=current_lang, sentiment=sentiment)
        print(f"Відповідь: {answer}")
        speak_multi_language(answer, lang=current_lang)

if __name__ == "__main__":
    main()