FROM python:3.8 as common

# Non-volative layers
WORKDIR /root
COPY requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt

COPY . /root/

ENTRYPOINT ["python", "main.py"]