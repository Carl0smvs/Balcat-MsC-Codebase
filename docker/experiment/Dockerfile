FROM python:3.10

ARG CI=False
ENV CI $CI

COPY Pipfile /
RUN apt update &&  \
    apt install ffmpeg libsm6 libxext6  -y && \
    pip install pipenv &&  \
    pipenv install  \
    && if [ "$CI" = "True" ]; then pipenv run pip install tensorflow[and-cuda]==2.15.1; fi

COPY experiment.py /


CMD ["pipenv", "run", "python", "experiment.py"]