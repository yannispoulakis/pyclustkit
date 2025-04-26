import torch
import os
import torch.nn as nn
from .._marcoge_utils.graph_classification_model import AverageMeter, GraphConvClassifier


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# train the model for one epoch or evaluation
def train_for_one_epoch(model, optimizer, loss_fn, dataloader, embeddings_dict, mode = 'train'):

    train_loss = AverageMeter('Loss', '.5f')
    num_correct_predictions = 0
    num_total = 0

    for batched_graph, labels, mapping in dataloader:

        batched_graph = batched_graph.to(device)
        node_emb = batched_graph.ndata['h']
        node_emb = node_emb.to(device)
        batched_graph.edata['weight'] = batched_graph.edata['weight'].to(device)
        labels = labels.to(device)

        if mode == 'train':
          optimizer.zero_grad()

        pred, emb = model(batched_graph,  node_emb)
        loss = loss_fn(pred, labels)

        for idx, lab, hidden in zip(mapping, labels.cpu().detach().tolist(), emb.cpu().detach().tolist()):
          hidden.append(int(lab))
          embeddings_dict[int(idx)] = hidden
        train_loss.update(loss.item(), len(labels))

        if mode == 'train':
          loss.backward()
          optimizer.step()

        #running_loss+=loss.item()

        num_correct_predictions += (pred.argmax(1) == labels).sum().item()
        num_total += len(labels)

    return round(num_correct_predictions*100/num_total,3), train_loss.avg, embeddings_dict


# train the model
def run_one_training(training_dataloader, test_dataloader, epochs, embeddings_dict):

    model = GraphConvClassifier(in_dim = 64,
                            hidden_dim = 300,
                            n_classes = 13)

    #device = torch.device("cuda:0")
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.006)

    for epoch in range(epochs):

      model.train()
      train_acc, train_loss, embeddings_dict = train_for_one_epoch(model, optimizer, loss_fn, training_dataloader, embeddings_dict, 'train')
      # print(f'Epoch {epoch} | \tTrain Accuracy: {train_acc:.2f}\tTrain Loss: {train_loss:.5f}', end ='\t')
      model.eval()
      test_acc, test_loss, embeddings_dict = train_for_one_epoch(model, optimizer, loss_fn, test_dataloader, embeddings_dict, 'eval')
      # print(f'Test Accuracy: {test_acc:.2f}\tTest Loss {test_loss:.5f}')
    #print('\n\n')
    torch.save(model, 'clustml/temp/gnn_model.pth')
    return model.state_dict(), embeddings_dict

