import json
import os

from datasets import Dataset, DatasetDict
from huggingface_hub import login


def count_tokens(text):
    """
    Count tokens from text (handles both string and list of sentences)
    For more accurate tokenization, you could use a proper tokenizer
    """
    if isinstance(text, list):
        # Join all sentences and then split on whitespace
        full_text = " ".join(text)
        return len(full_text.split())
    else:
        # Handle case where it's already a string
        return len(text.split())


def convert_list_to_string(text):
    """
    Convert list of sentences to a single string, removing <S> and </S> tags
    """
    if isinstance(text, list):
        # Join sentences and clean up tags
        full_text = " ".join(text)
        # Remove <S> and </S> tags
        full_text = full_text.replace("<S>", "").replace("</S>", "")
        # Clean up extra whitespace
        full_text = " ".join(full_text.split())
        return full_text
    else:
        # Already a string, just clean tags if present
        cleaned = text.replace("<S>", "").replace("</S>", "")
        return " ".join(cleaned.split())


def clean_data_file(input_file, output_file, min_tokens=100, max_tokens=300):
    """
    Clean data file by extracting only article_id and abstract_text fields
    and filtering by token length

    Args:
        input_file (str): Path to input file
        output_file (str): Path to output file
        min_tokens (int): Minimum number of tokens allowed (default: 100)
        max_tokens (int): Maximum number of tokens allowed (default: 300)
    """
    cleaned_data = []
    filtered_count = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            # Parse each line as JSON
            data = json.loads(line.strip())

            # Convert abstract to string if it's a list
            abstract_string = convert_list_to_string(data["abstract_text"])

            # Check token length
            abstract_tokens = count_tokens(abstract_string)

            if min_tokens <= abstract_tokens <= max_tokens:
                # Extract only the fields we want
                cleaned_entry = {
                    "article_id": data["article_id"],
                    "abstract_text": abstract_string,  # Now a clean string
                    "token_count": abstract_tokens,  # Optional: include token count
                }
                cleaned_data.append(cleaned_entry)
            else:
                filtered_count += 1

    # Write cleaned data to output file
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in cleaned_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    print(f"Kept {len(cleaned_data)} entries ({min_tokens}-{max_tokens} tokens)")
    print(
        f"Filtered out {filtered_count} entries (outside {min_tokens}-{max_tokens} token range)"
    )
    print(f"Output saved to: {output_file}")


def upload_to_huggingface(dataset_name, hf_token=None):
    """
    Upload cleaned datasets to Hugging Face Hub

    Args:
        dataset_name (str): Name for the dataset on HF Hub (e.g., "username/dataset-name")
        hf_token (str, optional): HF token for authentication
    """
    # Login to Hugging Face (you'll need to provide your token)
    if hf_token:
        login(token=hf_token)
    else:
        # This will prompt for token or use cached token
        login()

    # Load cleaned data files
    dataset_dict = {}
    splits = ["train", "test", "val"]

    for split in splits:
        cleaned_file = f"{split}_cleaned.txt"
        if os.path.exists(cleaned_file):
            data = []
            with open(cleaned_file, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line.strip()))

            # Convert to Hugging Face Dataset
            dataset_dict[split] = Dataset.from_list(data)
            print(f"Loaded {len(data)} examples for {split} split")
        else:
            print(f"Warning: {cleaned_file} not found, skipping {split} split")

    if dataset_dict:
        # Create DatasetDict
        dataset = DatasetDict(dataset_dict)

        # Upload to Hugging Face Hub
        dataset.push_to_hub(
            dataset_name, private=False
        )  # Set private=True for private dataset
        print(
            f"Successfully uploaded dataset to: https://huggingface.co/datasets/{dataset_name}"
        )
    else:
        print("No data files found to upload")


def main():
    # List of files to process
    files_to_process = ["train.txt", "test.txt", "val.txt"]

    # Set token range
    min_tokens = 100
    max_tokens = 300

    for file_name in files_to_process:
        if os.path.exists(file_name):
            # Create output filename (e.g., train.txt -> train_cleaned.txt)
            base_name = file_name.replace(".txt", "")
            output_file = f"{base_name}_cleaned.txt"

            try:
                clean_data_file(file_name, output_file, min_tokens, max_tokens)
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
        else:
            print(f"File {file_name} not found in current directory")

    # Upload to Hugging Face
    upload_choice = input("\nDo you want to upload to Hugging Face? (y/n): ").lower()
    if upload_choice == "y":
        dataset_name = input("Enter dataset name (format: username/dataset-name): ")
        hf_token = input(
            "Enter HF token (or press Enter to use cached/prompt): "
        ).strip()

        try:
            upload_to_huggingface(dataset_name, hf_token if hf_token else None)
        except Exception as e:
            print(f"Error uploading to Hugging Face: {str(e)}")
            print("Make sure you have the correct permissions and token")


if __name__ == "__main__":
    main()
