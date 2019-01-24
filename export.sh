export MODEL_NAME="ner"
export MODEL_BASE_PATH="/home/ycy/workingdir/bert0.1/bert/ner"
tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=ner --model_base_path=/home/ycy/workingdir/bert0.1/bert/ner
