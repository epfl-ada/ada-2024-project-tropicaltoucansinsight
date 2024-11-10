import os
import requests

def get_data(datasets, target_dir="data"):
    """
    Download the specified datasets from the web and save them in the target directory.
    
    :param datasets: (list of tuples) A list where each tuple contains is a pair of (URL, filename).
    :param target_dir: (str) The directory where the downloaded files will be saved.
    """
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    for dataset in datasets:
        # Ensure each dataset is a valid tuple
        if not isinstance(dataset, tuple) or len(dataset) != 2:
            print(f"Invalid dataset entry: {dataset}. It must be a tuple (URL, filename).")
            continue

        url, file_name = dataset
        file_path = os.path.join(target_dir, file_name)

        # Check if the file already exists
        if not os.path.exists(file_path):
            print(f"Downloading {file_name} from {url}...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Check if the request was successful

                with open(file_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            file.write(chunk)
                print(f"{file_name} downloaded successfully.")

            except requests.exceptions.RequestException as e:
                print(f"Error downloading {file_name}: {e}")

        else:
            print(f"{file_name} already exists in '{target_dir}'.")