FROM public.ecr.aws/lambda/python:3.9
RUN pip install keras-image-helper
RUN pip install --extra-index-url \
    https://googlecoral.github.io/py-repo/ tflite_runtime

COPY pneumoniadetector-model.tflite .
COPY lambda_func.py .
CMD ["lambda_func.lambda_handler"]

