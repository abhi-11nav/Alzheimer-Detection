FROM python:3.9
COPY . /Users/abhinav/Desktop/machinelearning/alzheimers_detection
EXPOSE 5000
WORKDIR /Users/abhinav/Desktop/machinelearning/alzheimers_detection
RUN pip install -r requirements.txt
CMD python app.py