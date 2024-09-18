from sagemaker.huggingface.model import HuggingFaceModel
import os

if __name__ == "__main__":

   # Grab necessary environment variables
   role = os.getenv("SAGEMAKER_ROLE")
   model_1 = os.getenv("MODEL_1") 
   entry_point = os.getenv("INFERENCE")

   # create Hugging Face Model Class
   huggingface_model = HuggingFaceModel(
      model_data=model_1, # path to fine-tuned model
      entry_point=entry_point, # path to inference file
      role=role,  
      transformers_version="4.37.0",                         
      pytorch_version="2.1.0",                               
      py_version='py310',                                    
   )
   # deploy model to SageMaker Inference
   predictor = huggingface_model.deploy(
      initial_instance_count=1,
      instance_type="ml.g4dn.xlarge",
      endpoint_name='finetuned-wav2vec-endpoint-spoof'
   )