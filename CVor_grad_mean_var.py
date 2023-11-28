import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

seed = 123

torch.manual_seed(seed)
np.random.seed(seed)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

# Initialize network, loss function
input_size, hidden_size, output_size = 10, 20, 1
model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss(reduction='none')

# Function to apply control variates
def apply_control_variates(gradients, control_variates, c_star):
    adjusted_gradients = []
    for grad, cv in zip(gradients, control_variates):
        # Log values to identify where NaN is originating
        print("Gradient:", grad)
        print("Control Variate:", cv)
        print("Mean Control Variate:", torch.mean(cv, dim=0))

        # Safeguard against NaN: Skip adjustment if NaN is found
        if torch.isnan(grad).any() or torch.isnan(cv).any():
            adjusted_grad = grad
        else:
            # Simplified control variate adjustment
            cv_mean = torch.mean(cv, dim=0)
            if not torch.isnan(cv_mean).any():
                adjusted_grad = grad + c_star * (cv - cv_mean)
            else:
                adjusted_grad = grad

        # Check and handle NaN in the adjusted gradient
        if torch.isnan(adjusted_grad).any():
            adjusted_grad = torch.nan_to_num(adjusted_grad)  # Replace NaN with 0

        adjusted_gradients.append(adjusted_grad)
    return adjusted_gradients


# Function to compute variance of gradients
def compute_variance_of_gradients(gradients):
    variances = []
    for grad in gradients:
        # Skip if the gradient is None
        if grad is not None:
            grad_flat = grad.view(-1)
            grad_var = torch.var(grad_flat)

            # Check if the variance is not NaN before appending
            if not torch.isnan(grad_var):
                variances.append(grad_var.item())

    # Handle the case where variances list might be empty
    if variances:
        average_variance = np.mean(variances)
    else:
        average_variance = float('nan')

    return average_variance


# Function to compute the RLOO control variate
def compute_rloo_control_variate(gradients):
    batch_size = gradients[0].shape[0]
    control_variates = []
    for grad in gradients:
        # Calculate RLOO control variate for each gradient
        rloo_cv = torch.stack([torch.mean(torch.cat([grad[:i], grad[i+1:]]), dim=0) for i in range(batch_size)])
        control_variates.append(rloo_cv)
    return control_variates

# Experiment
original_variances = []
adjusted_variances = []

def compute_mean_of_gradients(gradients):
    """ Compute the mean of gradients """
    means = [torch.mean(g).item() for g in gradients if g is not None]
    return np.mean(means) if means else float('nan')

# Initialize lists to store mean values
original_means = []
adjusted_means = []

def compute_rloo_loss(loss):
    """ Compute the RLOO version of loss """
    batch_size = loss.size(0)
    rloo_losses = []

    for i in range(batch_size):
        # Exclude the current element and calculate the mean of the rest
        excluded = torch.cat([loss[:i], loss[i+1:]])
        rloo_loss = excluded.mean()
        rloo_losses.append(rloo_loss)

    return torch.stack(rloo_losses)


import copy

def clone_model(model):
    return copy.deepcopy(model)

def cumulative_average(data):
    """Compute cumulative average of a list of values."""
    cumulative_sum = np.cumsum(data)
    iterations = np.arange(1, len(data) + 1)
    return cumulative_sum / iterations

def get_mean_grad_all_params(model):
    # Collect all gradients and flatten them
    all_grads = [param.grad.view(-1) for param in model.parameters() if param.grad is not None]
    # Concatenate all gradients to a single tensor
    all_grads_concat = torch.cat(all_grads)
    # Compute the mean of these gradients
    mean_grad = all_grads_concat.mean().item()
    return mean_grad

def get_param_grads(grads_list):
    """将梯度列表转换为每个参数的NumPy数组列表。"""
    num_params = len(grads_list[0])
    return [np.stack([grads[i].numpy() for grads in grads_list]) for i in range(num_params)]

# 存储原始梯度和CV调整后梯度的均值和方差
orig_means = []
cv_adj_means = []
orig_vars = []
cv_adj_vars = []

original_grads = []
cv_adjusted_grads = []

mean_diffs = []
variances = []

F = nn.Sequential(
    nn.Linear(output_size, output_size * 2),
    nn.Tanh(),
    nn.Linear(output_size * 2, output_size),
)

iteration = 500

for _ in range(iteration):
    X = torch.randn(50, input_size)
    Y = torch.randn(50, output_size)
    original_grads_1 = []
    cv_adjusted_grads_1 = []

    for i in range(X.size(0)):
        # 原始梯度计算
        output = model(X[i].unsqueeze(0))
        loss = criterion(output, Y[i].unsqueeze(0))
        model.zero_grad()
        loss.backward()
        original_grads_1.append([param.grad.clone() for param in model.parameters()])

        # Forward pass for one sample with CV adjustment
        output = model(X)
        loss = criterion(output, Y)

        # Apply your custom control variate process
        '''
        #########################################
        main part for CVor with NN control variate
        #########################################
        '''

        F_value = 0.15 * loss + F(loss)
        tilde_F_value = torch.exp(F_value - F_value.detach()).mean()
        CVor = torch.exp((torch.exp(tilde_F_value - tilde_F_value.detach())
                          - torch.exp(F_value - F_value.detach())))
        CVor_loss = CVor * loss

        '''
        #########################################
        '''

        # Backward pass for one sample (CV-adjusted gradient)
        model.zero_grad()
        CVor_loss[i].backward()
        cv_adjusted_grads_1.append([param.grad.clone() for param in model.parameters()])

    # 将梯度转换为每个参数的NumPy数组
    orig_grads_np = get_param_grads(original_grads_1)
    cv_adj_grads_np = get_param_grads(cv_adjusted_grads_1)

    # 对每个参数计算均值和方差
    orig_mean_vals = []
    cv_adj_mean_vals = []
    orig_var_vals = []
    cv_adj_var_vals = []
    for orig_grad, cv_adj_grad in zip(orig_grads_np, cv_adj_grads_np):
        # 计算每个参数的均值和方差
        orig_mean = np.mean(orig_grad, axis=0)
        cv_adj_mean = np.mean(cv_adj_grad, axis=0)
        orig_var = np.var(orig_grad, axis=0)
        cv_adj_var = np.var(cv_adj_grad, axis=0)

        # 添加每个参数的均值和方差
        orig_mean_vals.append(orig_mean)
        cv_adj_mean_vals.append(cv_adj_mean)
        orig_var_vals.append(orig_var)
        cv_adj_var_vals.append(cv_adj_var)

    # 存储每次迭代的均值和方差
    orig_means.append(np.mean([m.mean() for m in orig_mean_vals]))
    cv_adj_means.append(np.mean([m.mean() for m in cv_adj_mean_vals]))
    orig_vars.append(np.mean([v.mean() for v in orig_var_vals]))
    cv_adj_vars.append(np.mean([v.mean() for v in cv_adj_var_vals]))


orig_cum_means = np.cumsum(orig_means) / np.arange(1, iteration + 1)
cv_adj_cum_means = np.cumsum(cv_adj_means) / np.arange(1, iteration + 1)
orig_cum_vars = np.cumsum(orig_vars) / np.arange(1, iteration + 1)
cv_adj_cum_vars = np.cumsum(cv_adj_vars) / np.arange(1, iteration + 1)


# Creating a 2x2 subplot layout for four plots
plt.figure(figsize=(12, 12))

# Plot 1: Original Gradients Mean
plt.subplot(2, 2, 1)
plt.plot(orig_means, label='Original Gradients Mean')
plt.plot(cv_adj_means, label='CVor-Adjusted Gradients Mean')
plt.xlabel('Iteration')
plt.ylabel('Mean')
plt.title('Gradients Mean')
plt.legend()

# Plot 2: CV-Adjusted Gradients Mean
plt.subplot(2, 2, 2)
plt.plot(orig_vars, label='Original Gradients Variance')
plt.plot(cv_adj_vars, label='CVor-Adjusted Gradients Variance')
plt.xlabel('Iteration')
plt.ylabel('Mean')
plt.title('Gradients Variance')
plt.legend()

# Plot 3: Original Gradients Variance
plt.subplot(2, 2, 3)
plt.plot(orig_cum_means, label='Cumulative Mean of Original Gradients')
plt.plot(cv_adj_cum_means, label='Cumulative Mean of CV-Adjusted Gradients')
plt.xlabel('Iteration')
plt.ylabel('Cumulative Mean')
plt.title('Cumulative Mean of Gradients')
plt.legend()

# Plot 4: Cumulative Means
plt.subplot(2, 2, 4)
plt.plot(orig_cum_vars, label='Cumulative Variance of Original Gradients')
plt.plot(cv_adj_cum_vars, label='Cumulative Variance of CV-Adjusted Gradients')
plt.xlabel('Iteration')
plt.ylabel('Cumulative Variance')
plt.title('Cumulative Variance of Gradients')
plt.legend()

plt.tight_layout()
plt.show()