from tqdm import tqdm

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    correct = 0
    processed = 0
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(f'Loss={loss.item()} Accuracy={100*correct/processed:.2f}')
    return loss, correct / processed

