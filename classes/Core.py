class Core:
    def __init__(self):
        self.config = None  # Данные конфигурационного файла       
        self.db = None  # Объект подключения к БД
        self.req = None  # Запрос
        self.model = None  # Тут будем хранить модель
