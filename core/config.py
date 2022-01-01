from torch import device, cuda

class CFG:

    seed = 42
    #     learning_rate=3e-05
    max_epochs = 100
    batch_size = 8
    num_workers = 0
    precision = 16
    pin_memory = False

    device = device("cuda" if cuda.is_available() else "cpu")
    backbone = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
    strategy = "no_freeze"
    fast_dev_run = False

    train_file = "data/ourcalls_train_data_dir.csv"
    val_file = "data/ourcalls_val_data_dir.csv"
    test_file = "data/ourcalls_test_data_dir.csv"
    predict_file = "data/ourcalls_test_data_dir.csv"

    def _export_as_dict():
        return dict([i for i in __class__.__dict__.items() if i[0][:1] != "_"])
