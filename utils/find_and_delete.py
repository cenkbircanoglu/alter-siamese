import os

deleted_names = ["et.pkl",
                 "et_et_confusion_matrix_confusion.png",
                 "et_test_predictions.csv",
                 "et_train_predictions.csv",
                 "et_val_predictions.csv",
                 "nb_val_predictions.csv",
                 "nb.pkl",
                 "nb_confusion_matrix_confusion.png",
                 "nb_test_predictions.csv",
                 "nb_train_predictions.csv",
                 "voting.pkl",
                 "voting_test_predictions.csv",
                 "voting_train_predictions.csv",
                 "voting_val_predictions.csv",
                 "voting_voting_confusion_matrix_confusion.png",
                 "gb.pkl",
                 "gb_test_predictions.csv",
                 "gb_train_predictions.csv",
                 "gb_val_predictions.csv",
                 "gb_gb_confusion_matrix_confusion.png",
                 "svm_test_predictions.csv",
                 "svm_train_predictions.csv",
                 "svm_val_predictions.csv",
                 "test_predictions.csv",
                 "train_predictions.csv",
                 "val_predictions.csv",
                 "svm.pkl",
                 "mlp_test_predictions.csv",
                 "mlp_train_predictions.csv",
                 "mlp_val_predictions.csv",
                 "mlp.pkl",
                 "tsne.pdf",
                 "elastic_test_predictions.csv",
"elastic_train_predictions.csv",
"elastic_val_predictions.csv"
                 ]

path = "/media/cenk/2TB2/alter_siamese/results"
for root, directory, filenames in os.walk(path):
    for filename in filenames:
        if filename in deleted_names:
            deleted_name = os.path.join(root, filename)
            print(deleted_name)
            os.remove(deleted_name)

path = "/media/cenk/2TB2/alter_siamese/others"
for root, directory, filenames in os.walk(path):
    for filename in filenames:
        if filename in deleted_names:
            deleted_name = os.path.join(root, filename)
            print(deleted_name)
            os.remove(deleted_name)