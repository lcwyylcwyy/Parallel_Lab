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
    """Adafactor optimizer implemented in NumPy"""
    def __init__(self, params, lr=None, eps=(1e-30, 1e-3), clip_threshold=1.0, 
                 decay_rate=-0.8, beta1=None, weight_decay=0.0, scale_parameter=True, 
                 relative_step=True, warmup_init=False):
        self.params = params
        self.eps = eps
        self.clip_threshold = clip_threshold
        self.decay_rate = decay_rate
        self.beta1 = beta1
        self.weight_decay = weight_decay
        self.scale_parameter = scale_parameter
        self.relative_step = relative_step
        self.warmup_init = warmup_init
        self.lr = lr
        
        self.step_count = 0
        self.factored = []
        self.v_row = []
        self.v_col = []
        self.v = []
        self.m = []
        
        for param in params:
            # Determine if we can use factored second moment
            factored = len(param.shape) >= 2
            self.factored.append(factored)
            
            if factored:
                # For factored parameters, initialize row and column factors
                self.v_row.append(np.zeros(param.shape[0]))
                self.v_col.append(np.zeros(param.shape[1]))
                self.v.append(None)  # No full v needed for factored params
            else:
                # For non-factored parameters, initialize full v
                self.v_row.append(None)
                self.v_col.append(None)
                self.v.append(np.zeros_like(param))
                
            # Initialize momentum if beta1 is not None
            self.m.append(np.zeros_like(param) if beta1 is not None else None)
    
    def _get_lr(self):
        if self.lr is None:
            if self.relative_step:
                min_step = 1e-6 * self.step_count if self.warmup_init else 1e-2
                return min(min_step, 1.0 / np.sqrt(self.step_count))
            else:
                return 1e-3
        else:
            return self.lr
            
    def step(self, grads):
        """Performs a single optimization step."""
        self.step_count += 1
        lr = self._get_lr()
        
        for param_idx, param in enumerate(self.params):
            grad = grads[param_idx]
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param
                
            # Handle factorization for 2D+ tensors
            if self.factored[param_idx]:
                # Get parameter shapes
                shape = param.shape
                grad_sqr = np.square(grad)
                row_mean = grad_sqr.mean(axis=1)
                col_mean = grad_sqr.mean(axis=0)
                
                # Update running averages of the factored second moment
                decay_rate = 1.0 - np.power(self.step_count, self.decay_rate)
                self.v_row[param_idx] = decay_rate * self.v_row[param_idx] + (1 - decay_rate) * row_mean
                self.v_col[param_idx] = decay_rate * self.v_col[param_idx] + (1 - decay_rate) * col_mean
                
                # Calculate update using the factored second moment
                row_factor = 1.0 / np.sqrt(self.v_row[param_idx] + self.eps[0])
                col_factor = 1.0 / np.sqrt(self.v_col[param_idx] + self.eps[0])
                
                # Broadcast to match parameter shape
                row_factor = row_factor.reshape(-1, 1)
                update = grad * row_factor * col_factor
            else:
                # For non-factored params, update running average of second moment directly
                decay_rate = 1.0 - np.power(self.step_count, self.decay_rate)
                self.v[param_idx] = decay_rate * self.v[param_idx] + (1 - decay_rate) * np.square(grad)
                
                # Calculate update
                update = grad / np.sqrt(self.v[param_idx] + self.eps[0])
            
            # Apply momentum if beta1 is specified
            if self.beta1 is not None:
                self.m[param_idx] = self.beta1 * self.m[param_idx] + (1 - self.beta1) * update
                update = self.m[param_idx]
            
            # Apply clipping
            update_norm = np.linalg.norm(update.flatten())
            if update_norm > self.clip_threshold:
                update = update * self.clip_threshold / update_norm
            
            # Scale update
            if self.scale_parameter:
                update = update * lr
            else:
                update = update * lr * np.maximum(np.sqrt(self.eps[1]), param_scale)
            
            # Update parameter
            param -= update
    
    def zero_grad(self):
        pass
