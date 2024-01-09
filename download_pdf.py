import os
import urllib.request
from urllib.error import HTTPError, URLError

import certifi
import pandas as pd
import pdfplumber

os.environ["SSL_CERT_FILE"] = certifi.where()


def download_pdfs(data_df, destination_folder, failed_csv_path, timeout=60):
    failed_downloads = []  # List to store information about failed downloads
    data_df = data_df.dropna()

    for i, row in data_df.iterrows():
        url = row['URL'].replace(" ", "")
        document_id = row['ID']

        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/pdf, */*'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                pdf_path = os.path.join(destination_folder, f"{document_id}.pdf")
                with open(pdf_path, 'wb') as pdf_file:
                    pdf_file.write(response.read())
        except HTTPError as e:
            print(f"Failed to download PDF for ID {document_id}. HTTP Error: {e.code}")
            failed_downloads.append({'ID': document_id, 'Status Code': e.code, 'URL': url})
        except URLError as e:
            print(f"Failed to download PDF for ID {document_id}. URL Error: {e.reason}")
            failed_downloads.append({'ID': document_id, 'Status Code': None, 'URL': url})
        except Exception as e:
            print(f"Failed to download PDF for ID {document_id}. Error: {e}")
            failed_downloads.append({'ID': document_id, 'Status Code': None, 'URL': url})

    if failed_downloads:
        failed_df = pd.DataFrame(failed_downloads)
        failed_df.to_csv(failed_csv_path, index=False)
        print(f"Information about failed downloads saved to: {failed_csv_path}")

    return destination_folder



# Usage example
# destination_folder = 'path_to_save_pdfs'
# text_destination_folder = 'path_to_save_extracted_texts'
# failed_csv_path = 'path_to_save_failed_downloads.csv'
# download_pdfs_with_id(data_df, destination_folder, text_destination_folder, failed_csv_path, start_id='TP0')
