from sagemaker.huggingface import HuggingFace
import os
from sagemaker.debugger import TensorBoardOutputConfig

if __name__ == "__main__":

    # Grab necessary environment variables
    role = os.getenv("SAGEMAKER_ROLE")
    source = os.getenv("SOURCE")
    bucket = os.getenv("BUCKET")
    log_dir = os.getenv("LOG_DIR")

    training_input_path = f's3://{bucket}/train'
    test_input_path = f's3://{bucket}/test'
    output_path = f's3://{bucket}/output'

    # TensorBoard configuration
    tensorboard_output_config = TensorBoardOutputConfig(
        s3_output_path=os.path.join(f"s3://{bucket}", "output", 'tensorboard'),
        container_local_output_path=log_dir
        )

    # Define model hyperparameters
    hyperparameters = {'epochs': 5, 
                    'train_batch_size': 8,
                    'model_name': "facebook/wav2vec2-base-960h",
                    'training_dir': training_input_path,
                    'test_dir': test_input_path,
                    'output_dir': output_path,
                    'warmup_steps': 500,    # Add if you want to control warmup
                    }
                    
    # Define metrics definitions
    metric_definitions=[
        {'Name': 'loss', 
            'Regex': "'loss': ([0-9]+(.|e\-)[0-9]+),?"},
        {'Name': 'learning_rate',
            'Regex': "'learning_rate': ([0-9]+(.|e\-)[0-9]+),?"},
        {'Name': 'eval_loss',
            'Regex': "'eval_loss': ([0-9]+(.|e\-)[0-9]+),?"},
        {'Name': 'eval_accuracy',
            'Regex': "'eval_accuracy': ([0-9]+(.|e\-)[0-9]+),?"},
        {'Name': 'eval_precision',
            'Regex': "'eval_precision': ([0-9]+(.|e\-)[0-9]+),?"},
        {'Name': 'eval_recall',
            'Regex': "'eval_recall': ([0-9]+(.|e\-)[0-9]+),?"},
        {'Name': 'eval_runtime',
            'Regex': "'eval_runtime': ([0-9]+(.|e\-)[0-9]+),?"},
        {'Name': 'eval_samples_per_second',
            'Regex': "'eval_samples_per_second': ([0-9]+(.|e\-)[0-9]+),?"},
        {'Name': 'epoch', 
            'Regex': "'epoch': ([0-9]+(.|e\-)[0-9]+),?"}]
        
    # Create estimator 
    huggingface_estimator = HuggingFace(
        entry_point='train.py', 
        source_dir=source,
        output_path= output_path, 
        instance_type='ml.p3.2xlarge', # Must request access from AWS
        instance_count=1,
        transformers_version='4.36.0',
        pytorch_version='2.1.0',
        py_version='py310',
        role = role,
        hyperparameters = hyperparameters,
        metric_definitions = metric_definitions,
        tensorboard_output_config=tensorboard_output_config
        )

    # Start the training job
    huggingface_estimator.fit(
        {'train': training_input_path, 
        'test': test_input_path}
    )