from datasets import load_dataset, Audio
from transformers import Wav2Vec2FeatureExtractor
import s3fs

s3 = s3fs.S3FileSystem(anon=False,client_kwargs={'region_name': 'us-west-2'})
BUCKET = 'wav2vecproject'

model_str = "facebook/wav2vec2-base-960h" 

feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, # Defines the number of dimensions in the feature representation
        sampling_rate=16000, # Must be 16 kHz for Wav2Vec
        padding_value=0.0, # Pad sequences to the same length
        do_normalize=True, # Normalize audio data to a standard range
        return_attention_mask=False # Used for handling padding tokens
        )

def preprocess_function(samples):
    audio_arrays = [x["array"] for x in samples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=16000, # Must be 16 kHz for Wav2Vec
        return_tensors="pt", # Returns tensors in PyTorch format
        padding=True, # Pads sequences to the maximum length
        max_length=16000*8, # Maximum length of the sequences (8 seconds of audio at 16 kHz)
        do_normalize=True # Normalize audio data to a standard range
    )
    return inputs  

def extract_audio_array(sample):
    audio_array = sample["array"]
    return {"audio": audio_array}

def create_spoofed_dataset():
    # Load the csv dataset
    dataset = load_dataset(
        "csv", 
        data_files="datasets/spoof_genuine.csv", 
        split="train"
        )
    dataset = dataset.rename_column("classification", "label") 
    # Convert the 'label' values to binary format: 
    dataset = dataset.map(lambda x: 
                          {"label": 0 if x["label"] == "bonafide" else 1}) 
    # Cast the 'audio' column to the Audio type
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000)) 
    # Shuffle the dataset to randomize the order of samples
    dataset = dataset.shuffle(seed=42) 
    # Apply the 'preprocess_function' to the dataset in batches
    dataset = dataset.map( 
        preprocess_function, 
        batched=True, 
        remove_columns=["audio"], 
        batch_size=4
        )
        
    # Split the dataset into training (80%) and testing (20%) sets
    dataset = dataset.train_test_split(test_size=0.2) 
    # Save the two parts of the dataset
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    # Upload the training data to s3
    training_input_path = f's3://{BUCKET}/spoof_train/'
    train_dataset.save_to_disk(
        training_input_path,
        storage_options=s3.storage_options
        )

    # Upload the test data to s3
    test_input_path = f's3://{BUCKET}/spoof_test/'
    test_dataset.save_to_disk(
        test_input_path,
        storage_options=s3.storage_options
        )

def create_target_speaker_dataset(file_path):
    # Load the dataset and preprocess it
    dataset = load_dataset("csv", data_files=file_path, split="train") # Load the dataset from a CSV file
    dataset = dataset.rename_column("classification", "label") # Rename the 'classification' column to 'label' for clarity in the classification task
    dataset = dataset.map(lambda x: {"label": 0 if x["label"] == "target" else 1}) # Convert the 'label' values to binary format: 'target' becomes 0 and 'non-target' become 1
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000)) # Cast the 'audio' column to the Audio type
    dataset = dataset.shuffle(seed=42) # Shuffle the dataset to randomize the order of samples
    print(dataset)
    dataset = dataset.map( # Apply the 'preprocess_function' to the dataset in batches
        preprocess_function, 
        batched=True, 
        remove_columns=["audio"], 
        batch_size=4
        )
    dataset = dataset.train_test_split(test_size=0.2) # Split the dataset into training (80%) and testing (20%) sets
    print(dataset)

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # Save the training data to s3
    training_input_path = f's3://{BUCKET}/target_train/'
    train_dataset.save_to_disk(training_input_path,storage_options=s3.storage_options)

    # Save the test data to s3
    test_input_path = f's3://{BUCKET}/target_test/'
    test_dataset.save_to_disk(test_input_path,storage_options=s3.storage_options)
    print("EXITING")

def create_spoofed_eval_dataset(file_path):
    # Load the dataset and preprocess it
    dataset = load_dataset("csv", data_files=file_path) # Load the dataset from a CSV file
    dataset = dataset.rename_column("classification", "label") # Rename the 'classification' column to 'label' for clarity in the classification task
    dataset = dataset.map(lambda x: {"label": 0 if x["label"] == "bonafide" else 1}) # Convert the 'label' values to binary format: 'bonafide' becomes 0 and 'spoof' become 1
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000)) # Cast the 'audio' column to the Audio type
    dataset = dataset["train"]
    dataset = dataset.shuffle(seed=41) # Shuffle the dataset to randomize the order of samples
    dataset = dataset.map(extract_audio_array, input_columns=["audio"], remove_columns=["audio"])
     
    eval_path = f's3://{BUCKET}/eval_model1/'
    dataset.save_to_disk(eval_path,storage_options=s3.storage_options)

def create_target_eval_dataset(file_path):
    # Load the dataset and preprocess it
    dataset = load_dataset("csv", data_files=file_path) # Load the dataset from a CSV file
    dataset = dataset.rename_column("classification", "label") # Rename the 'classification' column to 'label' for clarity in the classification task
    dataset = dataset.map(lambda x: {"label": 0 if x["label"] == "target" else 1}) # Convert the 'label' values to binary format: 'target' becomes 0 and 'non-target' become 1
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000)) # Cast the 'audio' column to the Audio type
    dataset = dataset["train"]
    dataset = dataset.shuffle(seed=41) # Shuffle the dataset to randomize the order of samples
    dataset = dataset.map(extract_audio_array, input_columns=["audio"], remove_columns=["audio"])
     
    eval_path = f's3://{BUCKET}/eval_model2/'
    dataset.save_to_disk(eval_path,storage_options=s3.storage_options)


if __name__ == "__main__":

    # Call necessary functions; for example,
    file_path = "datasets/target_eval.csv"
    create_spoofed_dataset(file_path)
