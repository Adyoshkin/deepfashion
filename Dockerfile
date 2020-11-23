FROM python:3.6
ENV TZ=Europe/Moscow
ENV PYTHONUNBUFFERED=0
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
WORKDIR /root
ADD config.py .
ADD main.py .
ADD requirements.txt .
ADD best_model.hdf5 .
ADD dataset ./dataset
ADD imgs ./imgs
ADD network ./network
ADD scripts ./scripts
ADD static ./static
ADD templates ./templates
ADD utils ./utils
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r ./requirements.txt
ENTRYPOINT python main.py
EXPOSE 8008