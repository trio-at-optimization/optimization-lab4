from helper.package_dataset.module_mse import *


def train_torch_optim(
        f,
        x_train,
        y_train,
        eps_minimum,
        x0,
        loss_function,
        optimizer_method,
        num_epochs=10000,
        lr=0.01,
        momentum=None,
        nesterov=False,
        apply_min=True):

    # Определяем модель и исходные значения w
    w = torch.tensor(x0, requires_grad=True)

    if momentum is not None:
        if nesterov:
            optimizer = optimizer_method([w], lr=lr, momentum=momentum, nesterov=True)
        else:
            optimizer = optimizer_method([w], lr=lr, momentum=momentum)
    else:
        if nesterov:
            optimizer = optimizer_method([w], lr=lr, nesterov=True)
        else:
            optimizer = optimizer_method([w], lr=lr)

    points = [np.copy(w.detach().numpy())]
    # Цикл оптимизации
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Обнуляем градиенты

        # # Прямой проход
        # y_pred = f(x_train, w)

        # Вычисляем функцию потерь
        loss = loss_function(x_train, y_train, w, f)

        if apply_min and mse_loss_torch(x_train, y_train, w, f) < eps_minimum:
            break

        # Обратное распространение
        loss.backward()

        # Обновляем параметры w
        optimizer.step()

        # # Выводим прогресс оптимизации
        # if (epoch + 1) % 100 == 0:
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
        points.append(np.copy(w.detach().numpy()))

    # Получаем обученные значения w
    # trained_w = w.detach().numpy()
    return points
