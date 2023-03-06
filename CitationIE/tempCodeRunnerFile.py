
full_data_download_script = "full_data_downloads.sh"


def fetch_full_data_download_commands():
    full_data_download_commands = open(
        full_data_download_script).read().split("\n")
    full_data_download_commands = [
        x.split() for x in full_data_download_commands if len(x.split()) != 0]
    print(full_data_download_commands)
    return full_data_download_commands


fetch_full_data_download_commands()
