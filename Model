import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# --- Класс среды диабета ---
class DiabetesEnv(gym.Env):
    def __init__(self):
        super(DiabetesEnv, self).__init__()
        self.max_glucose = 400
        self.min_glucose = 40
        self.target_glucose = 110
        self.init_glucose = 180

        # Пространство наблюдений: [глюкоза, часы с последнего приёма пищи, количество углеводов в последней еде]
        self.observation_space = gym.spaces.Box(
            low=np.array([40, 0, 0]),
            high=np.array([400, 10, 100]),
            dtype=np.float32
        )
        # Действие: доза инсулина (0–10 ед/час)
        self.action_space = gym.spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32)

        self.max_dose_per_hour = 10.0
        self.min_hours_between_doses = 1
        self.last_dose_time = -100

        self.log = []
        self.reset()

    def reset(self):
        self.glucose = self.init_glucose + np.random.normal(0, 10)
        self.hours_since_meal = 0
        self.time = 0
        self.last_carbs = 0
        self.last_dose_time = -100
        self.log = []
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.glucose, self.hours_since_meal, self.last_carbs], dtype=np.float32)

    def step(self, action):
        # Логируем ТЕКУЩЕЕ состояние ПЕРЕД изменениями
        current_glucose = self.glucose
        current_time = self.time

        insulin = float(action[0])

        # Ограничения по частоте дозирования инсулина
        if insulin > self.max_dose_per_hour:
            insulin = self.max_dose_per_hour
        if (self.time - self.last_dose_time) < self.min_hours_between_doses:
            insulin = 0.0
        if insulin > 0:
            self.last_dose_time = self.time

        # Генерация еды с вероятностью в зависимости от времени суток
        if np.random.rand() < self._meal_probability(self.time % 24):
            carbs = np.random.uniform(20, 70)
            self.hours_since_meal = 0
        else:
            carbs = 0
            self.hours_since_meal += 1

        self.last_carbs = carbs

        # Логируем ПЕРЕД обновлением глюкозы
        # Теперь время, глюкоза, инсулин и углеводы соответствуют одному моменту
        self.log.append({
            "time": current_time,
            "glucose": current_glucose,  # глюкоза ДО воздействия
            "insulin": insulin,          # инсулин, который будет принять
            "carbs": carbs,             # углеводы, которые будут съедены
            "reward": 0                 # пока заглушка, обновим после расчета
        })

        # Физиология: еда поднимает, инсулин снижает глюкозу
        self.glucose += 0.7 * carbs
        self.glucose -= 7.0 * insulin
        self.glucose += np.random.normal(0, 3)
        self.glucose = np.clip(self.glucose, self.min_glucose, self.max_glucose)

        # Функция вознаграждения: максимизирует нахождение вблизи целевого уровня
        reward = -abs(self.glucose - self.target_glucose) / 30.0
        if self.glucose < 70:
            reward -= 5.0  # сильный штраф за гипогликемию
        elif self.glucose > 180:
            reward -= (self.glucose - 180) / 10.0  # прогрессивный штраф за гипергликемию

        # Обновляем reward в последней записи лога
        if self.log:
            self.log[-1]["reward"] = reward

        self.time += 1
        done = self.time >= 24 * 7  # эпизод = 7 суток
        return self._get_obs(), reward, done, {}

    def _meal_probability(self, hour):
        # Более вероятно есть в завтрак, обед и ужин
        if 7 <= hour <= 9 or 12 <= hour <= 14 or 18 <= hour <= 20:
            return 0.7
        return 0.1

    def get_log_dataframe(self):
        return pd.DataFrame(self.log)

def train_model():
    vec_env = make_vec_env(lambda: DiabetesEnv(), n_envs=4)
    model = PPO("MlpPolicy", vec_env, verbose=0, batch_size=64, learning_rate=2.5e-4, n_epochs=10)
    model.learn(total_timesteps=500_000)
    return model

# --- Генерация плана терапии после обучения ---
def simulate_plan(model):
    env = DiabetesEnv()
    obs = env.reset()
    for _ in range(24 * 7):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done:
            break
    return env.get_log_dataframe()

# --- Построение графика глюкозы ---
def plot_glucose(log_df):
    plt.figure(figsize=(15, 8))

    # График глюкозы
    plt.subplot(2, 1, 1)
    plt.plot(log_df["time"], log_df["glucose"], label="Глюкоза", linewidth=2)
    plt.axhline(70, color='red', linestyle='--', label='Гипогликемия')
    plt.axhline(180, color='orange', linestyle='--', label='Гипергликемия')
    plt.axhline(110, color='green', linestyle='-', alpha=0.7, label='Цель')
    plt.xlabel("Часы")
    plt.ylabel("Глюкоза (мг/дл)")
    plt.title("Траектория глюкозы за 7 дней")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # График инсулина и углеводов
    plt.subplot(2, 1, 2)
    plt.bar(log_df["time"], log_df["insulin"], alpha=0.7, label="Инсулин (ед)", color='blue')
    plt.bar(log_df["time"], log_df["carbs"]/10, alpha=0.7, label="Углеводы (г/10)", color='orange')
    plt.xlabel("Часы")
    plt.ylabel("Дозы")
    plt.title("Инсулин и углеводы")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# --- Расчёт метрик эффективности ---
def evaluate_metrics(df):
    # TIR — Time in Range (70–180 мг/дл), важно для клинической интерпретации
    glucose = df["glucose"].values
    tir = np.mean((glucose >= 70) & (glucose <= 180)) * 100
    hypo = np.mean(glucose < 70) * 100
    hyper = np.mean(glucose > 180) * 100

    print(f"\nМетрики терапии:")
    print(f"TIR (70–180): {tir:.1f}%")
    print(f"Гипогликемия (<70): {hypo:.1f}%")
    print(f"Гипергликемия (>180): {hyper:.1f}%")
    print(f"Средняя глюкоза: {glucose.mean():.1f} мг/дл")
    print(f"Стандартное отклонение: {glucose.std():.1f} мг/дл")

    # Дополнительные метрики
    total_insulin = df["insulin"].sum()
    total_carbs = df["carbs"].sum()
    print(f"\nОбщий инсулин за неделю: {total_insulin:.1f} ед")
    print(f"Общие углеводы за неделю: {total_carbs:.1f} г")

# --- Основной запуск ---
if __name__ == "__main__":
    print("Обучение модели...")
    model = train_model()
    print("Готово. Генерация терапии...")

    df = simulate_plan(model)
    plot_glucose(df)
    evaluate_metrics(df)

    # Формирование финального плана терапии
    df["day"] = df["time"] // 24
    df["hour"] = df["time"] % 24
    df_out = df[["day", "hour", "glucose", "carbs", "insulin"]].round(1)

    print("\nПлан терапии на первые 48 часов:")
    print(df_out.head(48))

    df.to_csv("therapy_plan_fixed.csv", index=False)
    print("\nСохранено в therapy_plan_fixed.csv")

    # Проверка корректности данных
    print("\nПроверка синхронизации данных:")
    print("Первые 10 записей:")
    print(df[["time", "glucose", "insulin", "carbs"]].head(10))
