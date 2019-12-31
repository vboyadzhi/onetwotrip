# One Two Trip Challenge

Ссылка на соревнование:
[https://boosters.pro/championship/onetwotrip_challenge](https://boosters.pro/championship/onetwotrip_challenge)

- по первой задаче 55 место (на 30.12.2019) 0.6768650
- по второй задаче 24 место (на 30.12.2019) 0.7368750

Проект создан при помощи `Kedro 0.15.5` 

Документация [https://kedro.readthedocs.io](https://kedro.readthedocs.io)

## Воспроизвести решение

### Установка зависимостей
Скачать данные соревнования
- `onetwotrip_challenge_sub1.csv`
- `onetwotrip_challenge_train.csv`
- `onetwotrip_challenge_test.csv`

и положить в папку `data/01_raw`

Из корня проекта исполнить:
```bash
python -m pip install kedro==0.15.5 #установка kedro 0.15.5
kedro install #установка библиотек из src/requirements.txt
```

### Запуск пайплайна Kedro

```bash
kedro run
```

### Cсылки

- Пайплайн [src/one_two_trip/pipeline.py](src/one_two_trip/pipeline.py)
- Ноды (исполняемые Python функции в Пайплайне) [src/one_two_trip/nodes/my_nodes.py](src/one_two_trip/nodes/my_nodes.py)
- Файлы для Сабмита [data/07_model_output](data/07_model_output)

## Working with Kedro from notebooks

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

## Ignoring notebook output cells in `git`

In order to automatically strip out all output cell contents before committing to `git`, you can run `kedro activate-nbstripout`. This will add a hook in `.git/config` which will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be left intact locally.

## Package the project

In order to package the project's Python code in `.egg` and / or a `.wheel` file, you can run:

```
kedro package
```

After running that, you can find the two packages in `src/dist/`.

## Building API documentation

To build API docs for your code using Sphinx, run:

```
kedro build-docs
```

See your documentation by opening `docs/build/html/index.html`.

## Building the project requirements

To generate or update the dependency requirements for your project, run:

```
kedro build-reqs
```

This will copy the contents of `src/requirements.txt` into a new file `src/requirements.in` which will be used as the source for `pip-compile`. You can see the output of the resolution by opening `src/requirements.txt`.

After this, if you'd like to update your project requirements, please update `src/requirements.in` and re-run `kedro build-reqs`.
