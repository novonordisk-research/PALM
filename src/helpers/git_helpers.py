import subprocess


def get_repo_name():
    try:
        remote_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
        repo_name = remote_url.decode("utf-8").strip().split("/")[-1].replace(".git", "")
        return repo_name
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
