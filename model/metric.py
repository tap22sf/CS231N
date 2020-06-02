import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct, len(target), correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def update_confusion_matrix(cm, output, target):
    with torch.no_grad():
        
        _, preds = torch.max(output, 1)
        for t, p in zip(target.cpu().view(-1), preds.cpu().view(-1)):
            cm[t.long(), p.long()] += 1

    return

def update_confusion_calc(cm):
    
    with torch.no_grad():
        
        # Calculate Sensitivity and PPV
        covid_tp    = cm[2,2] 
        covid_tn    = cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]
        covid_fp    = cm[0,2] + cm[1,2] 
        covid_fn    = cm[2,0] + cm[2,1] 
            
        ppv         = covid_tp / (covid_tp + covid_fp)
        npv         = covid_tn / (covid_tn + covid_fn)
        sens        = covid_tp / (covid_tp + covid_fn)
        spec        = covid_tn / (covid_fp + covid_tn)

    return ppv.numpy(), sens.numpy()