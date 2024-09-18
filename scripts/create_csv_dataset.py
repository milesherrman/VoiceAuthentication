import csv

def genuine_spoof(file_path):
    data = []
    j = 0
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                # extract speaker id, file name, attack type, and classification
                speaker_id = parts[0]
                classification = parts[4]
                if classification == "bonafide":
                    path = "/Volumes/Samsung USB/LA/TrainingGenuine/flac/" + parts[1] + ".flac"
                elif classification == "spoof":
                    path = "/Volumes/Samsung USB/LA/TrainingSpoofed/flac/" + parts[1] + ".flac"
                data.append([speaker_id, classification, path])
    return data

# Use some of the eval dataset to even out training classes
def equal_dataset_spoof():
    data = []
    file_path2 = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    file_path1 = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    bonafide_count = 0
    spoof_count = 0
    with open(file_path1, "r") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                # extract speaker id, file name, attack type, and classification
                classification = parts[4]
                if classification == "bonafide" and bonafide_count < 2500:
                    path = "LA/TrainingGenuine/flac/" + parts[1] + ".flac"
                    bonafide_count += 1
                    data.append([classification, path])
                elif classification == "spoof" and spoof_count < 13000:
                    path = "LA/TrainingSpoofed/flac/" + parts[1] + ".flac"
                    spoof_count += 1
                    data.append([classification, path])
    with open(file_path2, "r") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                # extract speaker id, file name, attack type, and classification
                classification = parts[4]
                if classification == "bonafide" and bonafide_count < 13000:
                    path = "LA/ASVspoof2019_LA_eval/flac/" + parts[1] + ".flac"
                    bonafide_count += 1
                    data.append([classification, path])
                
    return data

def create_spoof_csv(protocol_file_path, flac_file_directory):
    data = []
    with open(protocol_file_path, "r") as file:
        lines = file.readlines()
        for line in lines: 
            parts = line.split()
            if len(parts) >= 5:
                classification = parts[4]
                if classification == "bonafide":
                    path = flac_file_directory+ parts[1] + ".flac"
                    data.append([classification, path])
                elif classification == "spoof":
                    path = flac_file_directory + parts[1] + ".flac"
                    data.append([classification, path])
    return data

def eval_dataset_target():
    data = []
    file_path = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    bonafide_count = 0
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                # extract speaker id, file name, attack type, and classification
                speaker_id = parts[0]
                classification = parts[4]
                if classification == "bonafide" and bonafide_count < 300:
                    path = "LA/ASVspoof2019_LA_eval/flac/" + parts[1] + ".flac"
                    bonafide_count += 1
                    data.append(["non-target", path])
    return data

def target_non_target(file_path):
    data = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                # extract speaker id, file name, attack type, and classification
                speaker_id = parts[0]
                classification = parts[4]
                if classification == "bonafide":
                    path = "LA/TrainingGenuine/flac/" + parts[1] + ".flac"
                    data.append(["non-target", path])
    return data

# Scrape audio files for speaker 0035, for target/non-target model training
def speaker_35():
    data = []
    file_path = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    target = 0
    other = 0
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                # extract speaker id, file name
                speaker_id = parts[0]
                if speaker_id == "LA_0035" and target < 1500:
                    path = "/Volumes/Samsung USB/LA/ASVspoof2019_LA_eval/flac/" + parts[1] + ".flac"
                    target += 1
                    data.append(["target", path])
                elif other < 1500:
                    path = "/Volumes/Samsung USB/LA/ASVspoof2019_LA_eval/flac/" + parts[1] + ".flac"
                    other += 1
                    data.append(["non-target", path])
    return data

def save_to_csv(data, output_file):
    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["classification", "audio"])
        writer.writerows(data)


if __name__ == "__main__":
    # scrape the file and save to CSV
    data = speaker_35()
    output_file = "datasets/example_output.csv"
    save_to_csv(data, output_file)

