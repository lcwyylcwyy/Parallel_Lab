import numpy as np
from collections import defaultdict

class SGD:
    """Stochastic Gradient Descent optimizer implemented in NumPy"""
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr
        
    def step(self, grads):
        """Performs a single optimization step."""
        for param_idx, param in enumerate(self.params):
            param -= self.lr * grads[param_idx]
            
    def zero_grad(self):
        """Empty method to match PyTorch API, not needed for NumPy implementation"""
        pass

class SGDM:
    """SGD with Momentum implemented in NumPy"""
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(param) for param in params]
        
    def step(self, grads):
        """Performs a single optimization step."""
        for param_idx, param in enumerate(self.params):
            self.velocity[param_idx] = self.momentum * self.velocity[param_idx] + grads[param_idx]
            param -= self.lr * self.velocity[param_idx]
            
    def zero_grad(self):
        pass

class SGDM_Nesterov:
    """SGD with Nesterov Momentum implemented in NumPy"""
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(param) for param in params]

    def step(self, grads):
        """Performs a single optimization step with Nesterov momentum."""
        for param_idx, param in enumerate(self.params):
            # paper implementation
            # prev_velocity = self.velocity[param_idx].copy()
            # self.velocity[param_idx] = self.momentum * self.velocity[param_idx] + grads[param_idx]
            # nesterov_update = self.momentum * prev_velocity + grads[param_idx]
            # param -= self.lr * nesterov_update

            # pytorch implementation
            # Update velocity
            self.velocity[param_idx] = self.momentum * self.velocity[param_idx] + grads[param_idx]
            # Calculate effective gradient with Nesterov momentum (matches PyTorch implementation)
            effective_grad = grads[param_idx] + self.momentum * self.velocity[param_idx]
            # Update parameter
            param -= self.lr * effective_grad


    def zero_grad(self):
        pass

class Adagrad:
    """Adagrad optimizer implemented in NumPy"""
    def __init__(self, params, lr=0.01, eps=1e-10, initial_accumulator_value=0):
        self.params = params
        self.lr = lr
        self.eps = eps
        self.square_grads = [np.full_like(param, initial_accumulator_value) for param in params]
        
    def step(self, grads):
        """Performs a single optimization step."""
        for param_idx, param in enumerate(self.params):
            self.square_grads[param_idx] += np.square(grads[param_idx])
            adjusted_lr = self.lr / (np.sqrt(self.square_grads[param_idx]) + self.eps)
            param -= adjusted_lr * grads[param_idx]
            
    def zero_grad(self):
        pass

class RMSProp:
    """RMSProp optimizer implemented in NumPy"""
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8):
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.square_avg = [np.zeros_like(param) for param in params]
        
    def step(self, grads):
        """Performs a single optimization step."""
        for param_idx, param in enumerate(self.params):
            self.square_avg[param_idx] = self.alpha * self.square_avg[param_idx] + (1 - self.alpha) * np.square(grads[param_idx])
            param -= self.lr * grads[param_idx] / (np.sqrt(self.square_avg[param_idx]) + self.eps)
            
    def zero_grad(self):
        pass

class Adadelta:
    """Adadelta optimizer implemented in NumPy"""
    def __init__(self, params, rho=0.9, eps=1e-6):
        self.params = params
        self.rho = rho
        self.eps = eps
        self.square_avg = [np.zeros_like(param) for param in params]
        self.acc_delta = [np.zeros_like(param) for param in params]
        
    def step(self, grads):
        """Performs a single optimization step."""
        for param_idx, param in enumerate(self.params):
            self.square_avg[param_idx] = self.rho * self.square_avg[param_idx] + (1 - self.rho) * np.square(grads[param_idx])
            
            # Compute update
            std = np.sqrt(self.acc_delta[param_idx] + self.eps)
            delta = std * grads[param_idx] / np.sqrt(self.square_avg[param_idx] + self.eps)
            param -= delta
            
            # Update accumulation of parameter updates
            self.acc_delta[param_idx] = self.rho * self.acc_delta[param_idx] + (1 - self.rho) * np.square(delta)
            
    def zero_grad(self):
        pass

class Adam:
    """Adam optimizer implemented in NumPy"""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros_like(param) for param in params]
        self.v = [np.zeros_like(param) for param in params]
        self.t = 0
        
    def step(self, grads):
        """Performs a single optimization step."""
        self.t += 1
        for param_idx, param in enumerate(self.params):
            g = grads[param_idx]
            
            # Update biased first moment estimate
            self.m[param_idx] = self.betas[0] * self.m[param_idx] + (1 - self.betas[0]) * g
            
            # Update biased second raw moment estimate
            self.v[param_idx] = self.betas[1] * self.v[param_idx] + (1 - self.betas[1]) * np.square(g)
            
            # Bias correction
            m_hat = self.m[param_idx] / (1 - self.betas[0]**self.t)
            v_hat = self.v[param_idx] / (1 - self.betas[1]**self.t)
            
            # Update parameters
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
    def zero_grad(self):
        pass

class AdamW:
    """AdamW optimizer implemented in NumPy"""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(param) for param in params]
        self.v = [np.zeros_like(param) for param in params]
        self.t = 0
        
    def step(self, grads):
        """Performs a single optimization step."""
        self.t += 1
        for param_idx, param in enumerate(self.params):
            g = grads[param_idx]
            
            # Weight decay
            param -= self.lr * self.weight_decay * param
            
            # Update biased first moment estimate
            self.m[param_idx] = self.betas[0] * self.m[param_idx] + (1 - self.betas[0]) * g
            
            # Update biased second raw moment estimate
            self.v[param_idx] = self.betas[1] * self.v[param_idx] + (1 - self.betas[1]) * np.square(g)
            
            # Bias correction
            m_hat = self.m[param_idx] / (1 - self.betas[0]**self.t)
            v_hat = self.v[param_idx] / (1 - self.betas[1]**self.t)
            
            # Update parameters
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
    def zero_grad(self):
        pass

class Adafactor:
    """
    Adafactor optimizer implemented in NumPy
    
    Based on:
    "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost"
    https://arxiv.org/abs/1804.04235
    """
    def __init__(self, params, lr=None, eps1=1e-30, eps2=1e-3, 
                 clip_threshold=1.0, decay_rate=0.8, beta1=0.0,
                 weight_decay=0.0, relative_step=True, scale_parameter=True,
                 warmup_init=False):
        # Store parameters
        self.params = params
        self.lr = lr
        self.eps1 = eps1  # For stability in denominator
        self.eps2 = eps2  # Minimum learning rate
        self.clip_threshold = clip_threshold
        self.decay_rate = decay_rate
        self.beta1 = beta1
        self.weight_decay = weight_decay
        self.relative_step = relative_step
        self.scale_parameter = scale_parameter
        self.warmup_init = warmup_init
        
        # Initialize optimizer state
        self.step_count = 0
        self.m = [None] * len(params)  # First moment if beta1 > 0
        self.v_row = [None] * len(params)  # Factored 2nd moment (rows)
        self.v_col = [None] * len(params)  # Factored 2nd moment (columns)
        
        # Store shapes for factorization
        self.param_shapes = [p.shape for p in params]
    
    def _get_lr(self, param_idx):
        """Calculate the learning rate based on settings"""
        if not self.relative_step:
            # Use fixed learning rate
            return self.lr
            
        # Compute relative step size with minimum
        min_step = 1e-6 * self.step_count
        if self.warmup_init:
            lr = min(1.0, self.step_count / 10.0) * (0.1 / np.sqrt(min_step))
        else:
            lr = 1.0 / np.sqrt(min_step)
        
        # Scale by parameter scale if enabled
        if self.scale_parameter:
            param = self.params[param_idx]
            param_norm = np.sqrt(np.mean(np.square(param)) + self.eps1)
            lr = min(lr, 1.0 / param_norm)
        
        # Apply minimum learning rate
        return max(lr, self.eps2)
    
    def step(self, grads):
        """Perform one optimization step"""
        self.step_count += 1
        
        # Calculate beta2t (second moment decay rate) - stable calculation
        beta2t = 1.0 - (self.step_count ** (-self.decay_rate))
        beta2t = np.clip(beta2t, 0.0, 0.999)  # Clip for stability
        
        for param_idx, (param, grad) in enumerate(zip(self.params, grads)):
            if param.size == 0:
                continue
                
            # Get parameter shape
            shape = self.param_shapes[param_idx]
            
            # Apply momentum (first moment) if beta1 > 0
            if self.beta1 > 0.0:
                if self.m[param_idx] is None:
                    self.m[param_idx] = np.zeros_like(grad)
                
                self.m[param_idx] = self.beta1 * self.m[param_idx] + (1.0 - self.beta1) * grad
                grad_update = self.m[param_idx].copy()
            else:
                grad_update = grad.copy()
            
            # Apply factorized second moment estimation
            if len(shape) >= 2:
                # For matrices and higher-dimensional tensors
                
                # Reshape to 2D if needed (for tensors)
                if len(shape) > 2:
                    flat_shape = (shape[0], int(np.prod(shape[1:])))
                    grad_2d = grad.reshape(flat_shape)
                else:
                    grad_2d = grad
                
                # Initialize row/column statistics if needed
                if self.v_row[param_idx] is None:
                    self.v_row[param_idx] = np.zeros(grad_2d.shape[0])
                
                if self.v_col[param_idx] is None:
                    self.v_col[param_idx] = np.zeros(grad_2d.shape[1])
                elif len(self.v_col[param_idx]) != grad_2d.shape[1]:
                    # Reinitialize if shape mismatch
                    self.v_col[param_idx] = np.zeros(grad_2d.shape[1])
                    
                # Calculate squared gradients
                grad_sq = np.square(grad_2d)
                
                # Update row and column statistics
                row_mean = np.mean(grad_sq, axis=1)
                col_mean = np.mean(grad_sq, axis=0)
                
                # Update with EMA
                self.v_row[param_idx] = beta2t * self.v_row[param_idx] + (1.0 - beta2t) * row_mean
                self.v_col[param_idx] = beta2t * self.v_col[param_idx] + (1.0 - beta2t) * col_mean
                
                # Ensure non-negative values
                self.v_row[param_idx] = np.maximum(self.v_row[param_idx], 0.0)
                self.v_col[param_idx] = np.maximum(self.v_col[param_idx], 0.0)
                
                # Compute RMS values
                row_rms = np.sqrt(self.v_row[param_idx] + self.eps1).reshape(-1, 1)
                col_rms = np.sqrt(self.v_col[param_idx] + self.eps1).reshape(1, -1)
                
                # Compute normalization factor
                row_mean_rms = np.sqrt(np.mean(self.v_row[param_idx]) + self.eps1)
                
                # Calculate scaling factor for update
                scaling = (row_rms * col_rms) / np.maximum(row_mean_rms, 1e-5)
                
                # Reshape grad update if needed
                if len(shape) > 2:
                    update_2d = grad_update.reshape(flat_shape)
                else:
                    update_2d = grad_update
                
                # Scale the update
                update_2d = update_2d / np.maximum(scaling, 1e-5)
                
                # Reshape back to original shape if needed
                if len(shape) > 2:
                    update = update_2d.reshape(shape)
                else:
                    update = update_2d
            else:
                # For vectors (1D parameters)
                if self.v_row[param_idx] is None:
                    self.v_row[param_idx] = np.zeros_like(grad)
                
                # Update moment estimate
                self.v_row[param_idx] = beta2t * self.v_row[param_idx] + (1.0 - beta2t) * np.square(grad)
                
                # Ensure non-negative
                self.v_row[param_idx] = np.maximum(self.v_row[param_idx], 0.0)
                
                # Compute scaling
                scaling = np.sqrt(self.v_row[param_idx] + self.eps1)
                
                # Scale the update
                update = grad_update / np.maximum(scaling, 1e-5)
            
            # Apply update clipping if threshold is set
            if self.clip_threshold > 0.0:
                # Compute update norm
                update_norm = np.sqrt(np.sum(np.square(update)) + self.eps1)
                
                # Apply clipping if norm exceeds threshold
                if update_norm > self.clip_threshold:
                    update = update * (self.clip_threshold / update_norm)
            
            # Apply weight decay (decoupled)
            lr = self._get_lr(param_idx)
            if self.weight_decay > 0.0:
                param = param * (1.0 - lr * self.weight_decay)
            
            # Apply final update
            param -= lr * update
            
            # Final safety check for NaN/Inf values
            if np.isnan(param).any() or np.isinf(param).any():
                param = np.nan_to_num(param, nan=0.0, posinf=0.0, neginf=0.0)
    
    def zero_grad(self):
        """Empty method to match PyTorch API"""
        pass
