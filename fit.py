import time
from ModelTfrs import ModelTfrs


def fit(CORE):
    print('--- FIT MODEL ---')
    time_start = time.time()

    # Проверка наличия значения модели
    if 'model' not in CORE.req:
        return {'status': 'err', 'message': 'Bad request, model not found'}
    
    # Модели доступные по GET запросу 'model'
    if CORE.req['model'] == 'tfrs':
        CORE.model_tfrs = ModelTfrs()
        model_answer = CORE.model_tfrs.fit()
        print(f'MODEL ANSWER: {model_answer}')
    else:
        return {'status': 'err', 'message': 'Bad request, model not found'}


    time_delta = round(time.time() - time_start, 2)
    if model_answer:
        answer = {'status': 'OK', 'message': f'Data processed, execution time: {time_delta}s'}
    else:
        answer = {'status': 'ERROR'}

    return answer