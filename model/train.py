from datasets import load_from_disk
from transformers import Wav2Vec2ForSequenceClassification, Trainer, TrainingArguments, Wav2Vec2FeatureExtractor
import logging
import sys
import os
import argparse
import s3fs
import evaluate
from transformers.integrations import TensorBoardCallback

if __name__ == "__main__":

    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # Hyperparameters for training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=2e-5)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()


    # Load the datasets from S3
    s3 = s3fs.S3FileSystem()
    test_dataset = load_from_disk(dataset_path=args.test_dir, storage_options=s3.storage_options)
    train_dataset = load_from_disk(dataset_path=args.training_dir, storage_options=s3.storage_options)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")
    
    # Define Wav2Vec Feature Extractor
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, # Defines the number of dimensions in the feature representation
        sampling_rate=16000, # Must be 16 kHz for Wav2Vec
        padding_value=0.0, # Pad sequences to the same length
        do_normalize=True, # Normalize audio data to a standard range
        return_attention_mask=False # Used for handling padding tokens
        )

    # Define necessary metrics
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    # Compute metrics for evaluation
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        accuracy = accuracy_metric.compute(
            predictions=preds, references=labels)
        precision = precision_metric.compute(
            predictions=preds, references=labels)
        recall = recall_metric.compute(
            predictions=preds, references=labels)

        return {"accuracy": accuracy["accuracy"], 
                "precision": precision["precision"], 
                "recall": recall["recall"]}

    # Load the pretrained wav2vec 2.0 model for sequence classification
    model_name = "facebook/wav2vec2-base-960h"
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name, num_labels=2)

    # Define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,    
        evaluation_strategy="steps",
        eval_steps=40,
        save_strategy="no",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        fp16=True,  
        weight_decay=0.1,
        logging_steps=25,
        logging_dir='/home/sagemaker-user/SeniorProject/model/logs',
        warmup_steps=args.warmup_steps
    ) 

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[TensorBoardCallback()]
    )

    # Train model
    trainer.train()

    # Evaluate final model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # Write eval results to text file
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # Save the model to S3 bucket
    trainer.save_model(args.model_dir)


