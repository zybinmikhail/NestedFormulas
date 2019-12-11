import numpy as np

x_given = np.array([-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 0]).reshape(-1, 1)
length = x_given.shape[0]
X = np.concatenate([np.ones_like(x_given), x_given, x_given**2], axis=1)
y = np.array([1, 0.75, 0.5, 0.25, 0.25, 0.5, 0.75, 1]).reshape(-1,)
H = X @ np.linalg.inv(X.T @ X) @ X.T
threshold = 0.95

# Finding right point of interval using binary search
left = 0
right = 1000

while (right - left) >= 1e-5:
    middle = (left + right) / 2
    y_attempt = np.concatenate([y, [middle]])
    y_predict = H @ y_attempt
    losses = (y_attempt - y_predict)**2
    position_of_zero = np.argsort(np.argsort(losses))[length - 1]
    fraction_of_not_more_stranger_elements = (position_of_zero + 1) / (length)
    if fraction_of_not_more_stranger_elements <= threshold:
        left = middle
    else:
        right = middle
print(left)

# Finding left point of interval using binary search
left = -1000
right = 0

while (right - left) >= 1e-5:
    middle = (left + right) / 2
    y_attempt = np.concatenate([y, [middle]])
    y_predict = H @ y_attempt
    losses = (y_attempt - y_predict)**2
    position_of_zero = np.argsort(np.argsort(losses))[length - 1]
    fraction_of_not_more_stranger_elements = (position_of_zero + 1) / (length)
    if fraction_of_not_more_stranger_elements <= threshold:
        right = middle
    else:
        left = middle
print(left)