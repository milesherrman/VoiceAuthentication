from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
import s3fs
import torch
import os
import soundfile as sf

test_file = os.getenv("inference_file")

if __name__ == "__main__":
    
    s3 = s3fs.S3FileSystem()
    endpoint="finetuned-wav2vec-endpoint-spoof"
    predictor = Predictor(endpoint_name=endpoint, serializer=NumpySerializer())
    
    audio_data, sampling_rate = sf.read(test_file)
    input_json = {"audio_array": audio_data}
    with torch.no_grad():
        prediction = predictor.predict(data=input_json)
    print("Prediction:", prediction)
    