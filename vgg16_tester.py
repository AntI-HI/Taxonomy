from __future__ import print_function, division
# Ignore warnings
import warnings
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize([230, 230]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # =-========================22222222==========================================
    testset = ImageFolder(root='D:\\test', transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

    classes = ('cocacola_tin', 'icetea_lemon', 'icetea_peach' \
                   , 'nescafe_tin', 'nesfit', 'pepsi', 'pepsi_max' \
                   , 'pepsi_twist', 'redbull', 'redbull_sugar_free' \
                   , 'sprite', 'tadelle', 'tropicana_apricot' \
                   , 'tropicana_mixed', 'zuber')

    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision

    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()


    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        import matplotlib.pyplot as plt

        from sklearn.metrics import confusion_matrix
        import itertools
        import numpy as np
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


    device = torch.device("cuda")

    model = torchvision.models.vgg16()
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(
            in_features=2048,
            out_features=15
        ),
        torch.nn.Sigmoid()
    )
    model.load_state_dict(torch.load("model/vgg16.pth"))
    model.eval()
    model = model.cuda(device=device)

    class_correct = list(0. for i in range(15))
    class_total = list(0. for i in range(15))
    correct = 0
    total = 0
    with torch.no_grad():
        all_preds = torch.tensor([])
        targets = torch.tensor([], dtype=torch.long)
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)

            for i in labels:
                targets = targets.to(device)
                targets = torch.cat((targets, torch.tensor([i.item()]).to(device)), dim=0)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            outputs = outputs.to(device)
            all_preds = all_preds.to(device)
            all_preds = torch.cat(
                (all_preds, outputs)
                , dim=0
            )
            c = (predicted == labels).squeeze()
            for i in range(15):
                label = labels[i]

                class_correct[label] += c[i].item()
                class_total[label] += 1

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on test images: %d %%' % (
            100 * correct / total))
    for i in range(15):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    stacked = torch.stack(
        (
            targets
            , all_preds.argmax(dim=1)
        )
        , dim=1
    )

    cmt = torch.zeros(15, 15, dtype=torch.int32)

    for p in stacked:
        tl, pl = p.tolist()
        cmt[tl, pl] = cmt[tl, pl] + 1

    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import itertools
    import numpy as np

    cm = confusion_matrix(targets.cpu(), all_preds.argmax(dim=1).cpu())
    plot_confusion_matrix(cm, classes)

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    print('\nTRUE POSITIVES: ',TP)
    print('TRUE NEGATIVES: ',TN)
    print('FALSE POSITIVES: ',FP)
    print('FALSE NEGATIVES: ',FN)
    print('RECALL: ',TPR)
    print('ACC: ',ACC)
