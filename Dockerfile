# Базовый обрас Python 3.7.7 для Windows
#FROM stefanscherer/python-windows:nano
FROM python:3.7.7-buster

# Cоздание нового пользователя с именем ts_operator
#RUN adduser -D time_series

# Каталог по умолчанию, в котором будет установлено приложение
WORKDIR /home/time_series

# Передача файлов с компьютера в файловую систему контейнера
COPY requirements.txt requirements.txt
# Создание виртуальную среду
RUN python -m venv venv
RUN venv/bin/pip install -r requirements.txt

COPY static static
COPY templates templates
COPY autocorrelation.py figure_converter.py frac_diff.py fractal_difference.py fractal_dimension.py front_controller.py hurst.py ./

# Переменная среды внутри контейнера
ENV FLASK_APP front_controller.py

# Задать владельца всех каталогов и файлов
#RUN chown -R time_series:time_series ./
#USER time_series

EXPOSE 5000
ENTRYPOINT [ "python" ]

CMD [ "front_controller.py" ]