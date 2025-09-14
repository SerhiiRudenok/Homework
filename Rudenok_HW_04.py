import numpy as np
import matplotlib.pyplot as plt


"""
Завдання 1
Є дані про планові та фактичні продажі за місяцями. Потрібно порівняти два ряди на одному графіку.
•	Створіть список місяців і два списки значень: "План" і "Факт".
•	Побудуйте один графік із двома лініями.
•	Додайте підписи осей і заголовок.
•	Додайте легенду для ліній.
•	Відобразіть графік на екрані.
"""

months = ['Січ', 'Лют', 'Бер', 'Квіт', 'Трав', 'Черв', 'Лип', 'Серп', 'Вер', 'Жовт', 'Лист', 'Груд']

plan = np.array([120, 130, 125, 140, 150, 160, 155, 165, 170, 175, 180, 190])
fact = np.array([115, 128, 130, 135, 145, 158, 150, 160, 168, 170, 185, 195])

plt.figure(figsize=(10, 6))
plt.plot(months, plan, marker='o', label='План', color='blue')
plt.plot(months, fact, marker='s', label='Факт', color='green')

plt.xlabel('Місяць')
plt.ylabel('Продажі (тис. грн)')
plt.title('Завдання 1. Порівняння планових і фактичних продажів за місяцями')
plt.legend()
plt.grid(True)

plt.show()


"""
Завдання 2
Дано значення вікових груп 100 осіб. Потрібно показати розподіл.
•	Створіть список (або масив) зі 100 цілих значень віку.
•	Побудуйте гістограму розподілу.
•	Додайте вертикальну лінію середнього значення.
•	Підпишіть осі та додайте заголовок.
•	Відобразіть графік на екрані.
"""

np.random.seed(42)
ages = np.random.randint(18, 80, size=100)

plt.figure(figsize=(10, 6))
plt.hist(ages, bins=10, color='skyblue', edgecolor='black')

mean_age = np.mean(ages)
plt.axvline(mean_age, color='red', linestyle='dashed', linewidth=2, label=f'Середній вік: {mean_age:.1f}')

plt.xlabel('Вік')
plt.ylabel('Кількість осіб')
plt.title('Завдання 2. Розподіл віку серед 100 осіб')
plt.legend()
plt.grid(True)

plt.show()


"""
Завдання 3
Є результати іспитів за трьома групами студентів. Потрібно порівняти розкид оцінок між групами.
•	Створіть три набори числових результатів (по одній вибірці на групу).
•	Побудуйте графік boxplot для трьох груп поруч.
•	Підпишіть групи по осі X.
•	Додайте підписи осей і заголовок.
•	Відобразіть графік на екрані.
"""

np.random.seed(0)
group_A = np.random.randint(60, 100, size=30)
group_B = np.random.randint(50, 95, size=30)
group_C = np.random.randint(70, 100, size=30)

plt.figure(figsize=(10, 6))
plt.boxplot([group_A, group_B, group_C], tick_labels=['Група A', 'Група B', 'Група C'], patch_artist=True)

plt.xlabel('Група студентів')
plt.ylabel('Оцінки за іспит')
plt.title('Завдання 3. Порівняння розкиду оцінок між трьома групами')
plt.grid(True)

plt.show()


"""
Завдання 4
Є денні значення температури та вологості за тиждень. Потрібно показати обидві метрики на загальній діаграмі.
•	Створіть список дат (7 днів) і два списки значень: температура і вологість.
•	Побудуйте лінійний графік температури.
•	На тій самій області побудуйте графік вологості.
•	Додайте легенду, підписи осей і заголовок.
•	Поверніть підписи дат на осі X для читабельності та відобразіть графік.
"""

dates = ["Понеділок", "Вівторок", "Середа", "Четвер", "П'ятниця", "Субота", "Неділя"]

temperature = [22, 24, 23, 25, 26, 27, 24]
humidity = [60, 65, 63, 70, 72, 68, 66]

plt.figure(figsize=(10, 6))
plt.plot(dates, temperature, marker='o', label='Температура (°C)', color='tomato')
plt.plot(dates, humidity, marker='s', label='Вологість (%)', color='royalblue')

plt.xlabel('День тижня')
plt.ylabel('Значення')
plt.title('Завдання 4. Температура та вологість протягом тижня')
plt.legend()
plt.grid(True)

plt.xticks(rotation=90)

plt.show()


"""
Завдання 5
Є погодинні дані навантаження сервера за добу. Потрібно показати зміни і виділити область між кривою і віссю X.
•	Створіть список годин (0-23) і відповідні значення навантаження.
•	Побудуйте лінійний графік навантаження.
•	Зафарбуйте область під кривою.
•	Додайте сітку, підписи осей і заголовок.
•	Відобразіть графік на екрані.
"""

hours = np.arange(0, 24)

np.random.seed(1)
load = np.random.randint(20, 90, size=24)

plt.figure(figsize=(12, 6))
plt.plot(hours, load, marker='o', color='darkorange', label='Навантаження сервера')
plt.fill_between(hours, load, color='orange', alpha=0.3)

plt.xlabel('Година доби')
plt.ylabel('Навантаження (%)')
plt.title('Завдання 5. Погодинне навантаження сервера за добу')
plt.grid(True)
plt.legend()

plt.show()


"""
Завдання 6
Дано чотири набори метрик продукту: конверсія, утримання, середній чек, кількість замовлень. Потрібно показати їх на окремих панелях.
•	Створіть чотири набори числових значень за однією шкалою часу.
•	Створіть сітку з чотирьох підграфіків 2×2.
•	На кожному підграфіку побудуйте відповідний графік.
•	Додайте заголовки для кожного підграфіка та загальний заголовок.
•	Відобразіть результат на екрані.
"""

days = np.arange(1, 11)

np.random.seed(42)
conversion = np.random.uniform(0.05, 0.15, size=len(days)) * 100  # у %
retention = np.random.uniform(0.4, 0.8, size=len(days)) * 100     # у %
avg_check = np.random.randint(200, 500, size=len(days))           # грн
orders = np.random.randint(50, 150, size=len(days))               # кількість

fig, axs = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('Завдання 6. Метрики продукту за 14 днів', fontsize=16)

axs[0, 0].plot(days, conversion, color='teal', marker='o')
axs[0, 0].set_title('Конверсія (%)')
axs[0, 0].set_xlabel('День')
axs[0, 0].set_ylabel('%')
axs[0, 0].grid(True)

axs[0, 1].plot(days, retention, color='darkorange', marker='s')
axs[0, 1].set_title('Утримання (%)')
axs[0, 1].set_xlabel('День')
axs[0, 1].set_ylabel('%')
axs[0, 1].grid(True)

axs[1, 0].plot(days, avg_check, color='royalblue', marker='^')
axs[1, 0].set_title('Середній чек (грн)')
axs[1, 0].set_xlabel('День')
axs[1, 0].set_ylabel('грн')
axs[1, 0].grid(True)

axs[1, 1].plot(days, orders, color='green', marker='d')
axs[1, 1].set_title('Кількість замовлень')
axs[1, 1].set_xlabel('День')
axs[1, 1].set_ylabel('шт')
axs[1, 1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()