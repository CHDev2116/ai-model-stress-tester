import os


def prune_reports(report_dir, keep_latest=10):
    if not os.path.isdir(report_dir):
        return
    files = []
    for name in os.listdir(report_dir):
        path = os.path.join(report_dir, name)
        if os.path.isfile(path):
            files.append(path)
    files.sort(key=os.path.getmtime, reverse=True)
    for old_path in files[keep_latest:]:
        os.remove(old_path)
