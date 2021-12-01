def load_labels(file_path):
    kinetics_id_to_classname = {}
    with open(file_path, "r") as rf:
        for line in rf.readlines():
            id, classname = line.replace('\n', '').split(',')
            kinetics_id_to_classname[int(id)] = classname

    return kinetics_id_to_classname
