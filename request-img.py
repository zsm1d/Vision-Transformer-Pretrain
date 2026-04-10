import requests
import zipfile
import os



def download_zip(url, data_path, save_name=None):

    try:
        os.makedirs(data_path, exist_ok=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        if save_name is None:
            save_name = os.path.basename(url)
            if not save_name:
                save_name = "downloaded_file.zip"
            elif not save_name.endswith('.zip'):
                save_name += '.zip'
        
        save_path_name = os.path.join(data_path, save_name)

        with open(save_path_name, 'wb') as f:
            print("Downloading the file...")
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"The file is downloaded and saved as: {save_path_name}")
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"Download is fail: {e}")
        return False

def unzip_file(zip_path, extract_to=None):
    try:
        if extract_to is None:
            extract_to = "dataset"
        
        os.makedirs(extract_to, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print("Unzipping the zipfile...")
            zip_ref.extractall(extract_to)
        
        print(f"The zip file is unzipped at: {os.path.abspath(extract_to)}")
        return True
    
    except FileNotFoundError:
        print(f"Error: Cannot find the zipfile '{zip_path}'")
        return False
    except zipfile.BadZipFile:
        print(f"Error: '{zip_path}' is not a valid zipfile")
        return False
    except Exception as e:
        print(f"The unzip is fail: {e}")
        return False

if __name__ == "__main__":
    zip_url = input("Enter file URL: ").strip()
    if not zip_url:
        zip_url = "https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/resolve/main/images.zip"
    
    custom_name = input("Enter the saved name (Empty for the name in the URL): ").strip()
    if not custom_name:
        custom_name = None

    data_path = input("Enter the saved dir name (default is (dataset)): ").strip()
    if not data_path:
        data_path = "dataset"
    
    download_zip(zip_url, data_path, custom_name)

    zip_path = input("Enter the zip path: ").strip()
    if not zip_path:
        zip_path = "images.zip"
    
    extract_to = input("Enter the unzip dir (Default is (dataset)): ").strip()
    if not extract_to:
        extract_to = "dataset" 
    
    unzip_file(zip_path, extract_to)