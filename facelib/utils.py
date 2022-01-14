from tqdm import tqdm
import requests
import os

def download_weight(link, file_name, verbose=True):
    response = requests.get(link, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, 
                        desc='downloading defualt weights',
                        disable=False if verbose else True)
    
    with open(file_name, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        exit("ERROR, something went wrong (check your connection)")
        
