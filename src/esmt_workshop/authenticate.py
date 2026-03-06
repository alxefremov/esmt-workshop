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
        from google.collab import data_table
        data_table.enable_dataframe_formatter()

    result = subprocess.run(
        ["gcloud", "config", "get-value", "account"],
        capture_output=True, text=True
    )
    gcloud_email = result.stdout.strip()
    print(f"Authenticated as: {gcloud_email if gcloud_email else 'Unknown'}")
    os.environ['WORKSHOP_EMAIL'] = gcloud_email if gcloud_email else os.environ.get('WORKSHOP_EMAIL', '')

    # Students only provide email; proxy endpoint details are managed by organizers.
    student_email = os.getenv('WORKSHOP_EMAIL', 'student@example.com')
    print('STUDENT_EMAIL =', student_email)

    return student_email
