import torch

# Simple Linear Layer -  Computation Graph
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print(loss)

# Gradient function for parameters and loss function
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# Compute gradient
#We can only perform gradient calculations using backward once on a given graph, for performance reasons.
#  If we need to do several backward calls on the same graph, we need to pass retain_graph=True to the backward call.
loss.backward()
print(w.grad)
print(b.grad)

# Disable gradient tracking, when:
#To mark some parameters in your neural network as frozen parameters.
#To speed up computations when you are only doing forward pass, because computations on tensors that do not track gradients would be more efficient.
z = torch.matmul(x, w)+b
print(z.requires_grad)

# Methods 1 - no_grad
with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# Method 2 - detach
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

