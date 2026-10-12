{\rtf1\ansi\ansicpg936\cocoartf2870
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import copy\
import torch\
\
\
def train_model(\
    model,\
    optimizer,\
    trainloader,\
    criterion,\
    num_epochs=5\
):\
    loss_history = []\
\
    for epoch in range(num_epochs):\
        model.train()\
        running_loss = 0.0\
\
        for images, labels in trainloader:\
            optimizer.zero_grad()\
\
            outputs = model(images)\
            loss = criterion(outputs, labels)\
\
            loss.backward()\
            optimizer.step()\
\
            running_loss += loss.item()\
\
        average_loss = running_loss / len(trainloader)\
        loss_history.append(average_loss)\
\
        print(\
            f"Epoch \{epoch + 1\}/\{num_epochs\}, "\
            f"Loss: \{average_loss:.4f\}"\
        )\
\
    return loss_history\
\
\
def evaluate_model(model, testloader):\
    correct = 0\
    total = 0\
\
    model.eval()\
\
    with torch.no_grad():\
        for images, labels in testloader:\
            outputs = model(images)\
\
            _, predicted = torch.max(outputs, 1)\
\
            total += labels.size(0)\
            correct += (predicted == labels).sum().item()\
\
    accuracy = 100 * correct / total\
\
    return accuracy\
\
\
def train_with_validation(\
    model,\
    optimizer,\
    trainloader,\
    testloader,\
    criterion,\
    num_epochs=20\
):\
    train_loss_history = []\
    test_loss_history = []\
    train_acc_history = []\
    test_acc_history = []\
\
    for epoch in range(num_epochs):\
\
        model.train()\
\
        running_loss = 0.0\
        correct = 0\
        total = 0\
\
        for images, labels in trainloader:\
            optimizer.zero_grad()\
\
            outputs = model(images)\
            loss = criterion(outputs, labels)\
\
            loss.backward()\
            optimizer.step()\
\
            running_loss += loss.item()\
\
            _, predicted = torch.max(outputs, 1)\
\
            total += labels.size(0)\
            correct += (predicted == labels).sum().item()\
\
        train_loss = running_loss / len(trainloader)\
        train_accuracy = 100 * correct / total\
\
        model.eval()\
\
        test_running_loss = 0.0\
        test_correct = 0\
        test_total = 0\
\
        with torch.no_grad():\
            for images, labels in testloader:\
                outputs = model(images)\
                loss = criterion(outputs, labels)\
\
                test_running_loss += loss.item()\
\
                _, predicted = torch.max(outputs, 1)\
\
                test_total += labels.size(0)\
                test_correct += (\
                    predicted == labels\
                ).sum().item()\
\
        test_loss = test_running_loss / len(testloader)\
        test_accuracy = 100 * test_correct / test_total\
\
        train_loss_history.append(train_loss)\
        test_loss_history.append(test_loss)\
        train_acc_history.append(train_accuracy)\
        test_acc_history.append(test_accuracy)\
\
        print(\
            f"Epoch \{epoch + 1\}/\{num_epochs\} | "\
            f"Train Loss: \{train_loss:.4f\} | "\
            f"Test Loss: \{test_loss:.4f\} | "\
            f"Train Acc: \{train_accuracy:.2f\}% | "\
            f"Test Acc: \{test_accuracy:.2f\}%"\
        )\
\
    return (\
        train_loss_history,\
        test_loss_history,\
        train_acc_history,\
        test_acc_history\
    )\
\
\
def train_with_early_stopping(\
    model,\
    optimizer,\
    trainloader,\
    testloader,\
    criterion,\
    num_epochs=50,\
    patience=5\
):\
    train_loss_history = []\
    test_loss_history = []\
    train_acc_history = []\
    test_acc_history = []\
\
    best_test_loss = float("inf")\
    best_model_state = None\
    patience_counter = 0\
\
    for epoch in range(num_epochs):\
\
        model.train()\
\
        running_loss = 0.0\
        correct = 0\
        total = 0\
\
        for images, labels in trainloader:\
            optimizer.zero_grad()\
\
            outputs = model(images)\
            loss = criterion(outputs, labels)\
\
            loss.backward()\
            optimizer.step()\
\
            running_loss += loss.item()\
\
            _, predicted = torch.max(outputs, 1)\
\
            total += labels.size(0)\
            correct += (predicted == labels).sum().item()\
\
        train_loss = running_loss / len(trainloader)\
        train_accuracy = 100 * correct / total\
\
        model.eval()\
\
        test_running_loss = 0.0\
        test_correct = 0\
        test_total = 0\
\
        with torch.no_grad():\
            for images, labels in testloader:\
                outputs = model(images)\
                loss = criterion(outputs, labels)\
\
                test_running_loss += loss.item()\
\
                _, predicted = torch.max(outputs, 1)\
\
                test_total += labels.size(0)\
                test_correct += (\
                    predicted == labels\
                ).sum().item()\
\
        test_loss = test_running_loss / len(testloader)\
        test_accuracy = 100 * test_correct / test_total\
\
        train_loss_history.append(train_loss)\
        test_loss_history.append(test_loss)\
        train_acc_history.append(train_accuracy)\
        test_acc_history.append(test_accuracy)\
\
        print(\
            f"Epoch \{epoch + 1\}/\{num_epochs\} | "\
            f"Train Loss: \{train_loss:.4f\} | "\
            f"Test Loss: \{test_loss:.4f\} | "\
            f"Train Acc: \{train_accuracy:.2f\}% | "\
            f"Test Acc: \{test_accuracy:.2f\}%"\
        )\
\
        if test_loss < best_test_loss:\
            best_test_loss = test_loss\
            best_model_state = copy.deepcopy(\
                model.state_dict()\
            )\
            patience_counter = 0\
        else:\
            patience_counter += 1\
\
        if patience_counter >= patience:\
            print(\
                f"Early stopping triggered "\
                f"at epoch \{epoch + 1\}"\
            )\
            break\
\
    if best_model_state is not None:\
        model.load_state_dict(best_model_state)\
\
    return (\
        train_loss_history,\
        test_loss_history,\
        train_acc_history,\
        test_acc_history\
    )\
}