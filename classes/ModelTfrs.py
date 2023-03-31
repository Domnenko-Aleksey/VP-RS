import os
import requests
import pandas as pd
import numpy as np
from typing import Dict, Text

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs


class ModelTfrs():
    def __init__(self):
        self.data_path = 'files/data/video_dataset.csv'
        self.popular_list = []  # Тут будет храниться список 1000 популярных каналов
        self.embedding_dimension = 64  # Размер сжатого признакового пространства


    # === ОБУЧЕНИЕ МОДЕЛИ ===
    def fit(self):
        print('--- MODEL TFRS .fit() ---')

        # --- ПОЛУЧЕНИЕ ДАННЫХ ---
        # В данном варианте у нас нет данных из БД, поэтому читаем те, что есть из файла

        df = pd.read_csv(self.data_path)

        # Отбираем 1000 популярных видео по взаимодействию
        df_1000 = df.groupby(['video_id']).size().sort_values(ascending=False)[0:1000]
        df_1000.columns = ['video_id', 'count']

        # Отбираем записи
        reducer = df['video_id'].isin(df_1000.index)
        df_2 = df[reducer]
        print(f'Строк до фильтрации: {df.shape[0]}, после: {df_2.shape[0]}')

        # Получаем tf датасет уникальных пользователей
        users_id_arr = df['user_id'].astype('bytes').unique()

        # Получаем tf датасет уникальных видео
        videos_id_arr = df['video_id'].astype('bytes').unique()
        self.video_id_tensor = tf.data.Dataset.from_tensor_slices(videos_id_arr)

        # Получаем tf датасет ratings
        df_ratings = df[['user_id', 'video_id']]
        ratings_b_arr = df_ratings.to_numpy().astype('bytes')
        ratings = tf.data.Dataset.from_tensor_slices({"video_id": ratings_b_arr[:, 1], "user_id": ratings_b_arr[:, 0]})

        # Перемешиваем выборку
        tf.random.set_seed(42)
        shuffled = ratings.shuffle(df_ratings.shape[0], seed=42, reshuffle_each_iteration=False)

        # Разбиваем выборку на учебную и тестовую
        train_num = 10000
        test_num = 2000
        train = shuffled.take(train_num)
        test = shuffled.skip(train_num).take(test_num)

        # --- MODEL ---
        # Башня запросов
        user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=users_id_arr, mask_token=None),
            tf.keras.layers.Embedding(len(users_id_arr) + 1, self.embedding_dimension)
        ])

        # Башня кандидатов
        movie_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=videos_id_arr, mask_token=None),
            tf.keras.layers.Embedding(len(videos_id_arr) + 1, self.embedding_dimension)
        ])

        # Метрики
        # Вычисляет показатели для K лучших кандидатов, обнаруженных моделью поиска
        metrics = tfrs.metrics.FactorizedTopK(
        # Применяем к батчу данных chanells_tensor нашу модель кандидата
            candidates = self.video_id_tensor.batch(1000).map(movie_model)
        )

        # Ошибки
        task = tfrs.tasks.Retrieval(
            metrics=metrics
        )

        model = CreateModel(user_model, movie_model, task)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

        cached_train = train.shuffle(100_000).batch(32).cache()
        cached_test = test.batch(32).cache()

        # Обучим модель
        history = model.fit(cached_train, epochs=3)

        # Оценка модели
        # model.evaluate(cached_test, return_dict=True)

        # Устанавливаем модель
        self.model = model

        return True


    # === РЕКОМНДОВАННЫЕ КАНАЛЫ ===
    # Предсказание - переопределяем в наследуемом классе
    def predict(self, user_id):
        # Создаём модель, которая использует необработанные функции запроса, и
        index = tfrs.layers.factorized_top_k.BruteForce(self.model.user_model)
        # рекомендует фильмы из всего набора данных фильмов.
        index.index_from_dataset(
            tf.data.Dataset.zip((self.video_id_tensor.batch(10000), self.video_id_tensor.batch(10000).map(self.model.movie_model)))
        )

        # Получить рекомендации для пользователя с индексом user_id, например: "38082".
        _, titles = index(tf.constant([user_id]))
        answer = titles.numpy().astype('str').tolist()

        print(f'---ANSWER: {answer}---')

        return answer[0]


# ======= СОЗДАЁМ МОДЕЛЬ =======
class CreateModel(tfrs.Model):
    def __init__(self, user_model, movie_model, task):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # Выбираем пользовательские функции и передаем их в пользовательскую модель.
        user_embeddings = self.user_model(features["user_id"])
        # И выберите функции фильма и передайте их в модель фильма, вернуть вложения.
        positive_movie_embeddings = self.movie_model(features["video_id"])

        # Вычисляет потери и метрики.
        return self.task(user_embeddings, positive_movie_embeddings)
