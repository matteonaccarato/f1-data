import io
import zipfile

import requests

gh_latest_release = 'https://api.github.com/repos/f1db/f1db/releases/latest'

folder = 'f1db-csv'
last_version_file = "f1db_last_version_file.txt"

# Download and extract from {url} to {folder}
def download(url, last_version):
    print(f"f1-data > Downloading\t\t\t({last_version})")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    zf.extractall(folder)
    with open(last_version_file, 'w') as file:
        file.write(last_version)
        print(f"f1-data > Successfully Downloaded\t({last_version})")
    return {
        "ok": True,
        "updated": True,
        "version": last_version,
        "message": f"Successfully downloaded {last_version}."
    }

# Get Data only if a new version is available
def get_data():
    try:
        response = requests.get(gh_latest_release, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as err:
        return {
            "ok": False,
            "updated": False,
            "version": None,
            "message": f"Unable to fetch latest release: {err}"
        }

    if "tag_name" not in data:
        return {
            "ok": False,
            "updated": False,
            "version": None,
            "message": "Latest release payload does not include a version name."
        }

    last_version = data["tag_name"]
    print(f"f1-data > Fetching Version\t\t({last_version})")
    url = f'https://github.com/f1db/f1db/releases/download/{last_version}/f1db-csv.zip'

    try:
        with open(last_version_file, 'r') as file:
            content = file.read().strip()

        if content == last_version:
            print(f"f1-data > Already up to date\t({last_version})")
            return {
                "ok": True,
                "updated": False,
                "version": last_version,
                "message": f"Already up to date ({last_version})."
            }
        return download(url, last_version)
    except FileNotFoundError:
        return download(url, last_version)
    except (requests.RequestException, zipfile.BadZipFile, OSError) as err:
        return {
            "ok": False,
            "updated": False,
            "version": last_version,
            "message": f"Unable to update data: {err}"
        }