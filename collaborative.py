# Получаем рекомендации для пользователя
def collaborative(CORE):
    # Проверка наличия значения модели
    if 'user_id' not in CORE.req:
        return {'status': 'err', 'message': 'Bad request, user_id not found'}

    user_id = CORE.req['user_id']

    video_id = CORE.model_tfrs.predict(user_id)  # Получаем данные по API
    return {'status': 'ok', 'user_id': user_id, 'video_id': video_id}