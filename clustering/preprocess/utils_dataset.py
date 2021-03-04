import json

def read_data(base_dir, target_user):
    t_emails = []
    #t_labels = []

    print(base_dir + target_user + ".json")

    with open(base_dir + target_user + ".json") as f:
        mails = json.load(f)
    for mail in mails["sent"]:
        if(mail["body"] == ""):
            continue
        to_clean = mail["body"]
        clean = to_clean.replace('\r', ' ')
        clean = to_clean.replace('\n', ' ')
        t_emails.append(clean)
        # t_labels.append(1)

    return t_emails
