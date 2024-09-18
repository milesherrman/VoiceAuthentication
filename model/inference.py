import logging
import json
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, 
    sampling_rate=16000, 
    padding_value=0.0, 
    do_normalize=True, 
    return_attention_mask=False
    )
    
logger.info("Libraries are loaded")

def preprocess_function(input_data):
    inputs = feature_extractor(
        input_data,
        sampling_rate=16000, # Must be 16 kHz for Wav2Vec
        return_tensors="pt", # Returns tensors in PyTorch format
        padding=True, # Pads sequences to the maximum length
        max_length=16000*8, # Maximum length of the sequences (8 seconds of audio at 16 kHz)
        do_normalize=True # Normalize audio data to a standard range
    )
    return inputs  

def model_fn(model_dir, context=None):
        device = get_device()
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir).to(device) 
                
        return model

def input_fn(json_request_data, content_type='application/json'): 
    device = get_device()
    input_data = json_request_data['audio_array']
    inputs = preprocess_function(input_data).to(device)
    
    return inputs

def predict_fn(input_data, model):
    with torch.no_grad():
        logits = model(input_data["input_values"]).logits
    prediction = torch.argmax(logits, dim=-1).item()

    return prediction
    
def output_fn(predictions, accept='application/json'):    
    return json.dumps(predictions)

def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device
