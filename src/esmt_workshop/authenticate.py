import os
import subprocess

try:
    from google.colab import auth
    _ON_COLAB = True
except ImportError:
    _ON_COLAB = False


def authenticate():
    if _ON_COLAB:
        auth.authenticate_user()

    result = subprocess.run(
        ["gcloud", "config", "get-value", "account"],
        capture_output=True, text=True
    )
    gcloud_email = result.stdout.strip()
    print(f"Authenticated as: {gcloud_email if gcloud_email else 'Unknown'}")
    os.environ['WORKSHOP_EMAIL'] = gcloud_email if gcloud_email else os.environ.get('WORKSHOP_EMAIL', '')
