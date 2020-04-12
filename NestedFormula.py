import copy
import torch
import torch.nn as nn
import torchcontrib
from auxiliary_functions import *

class NestedFormula(nn.Module):
    """
    Class used for representing formulas
    
    Attributes:
        depth
        num_variables
        subformulas - list of subformulas of smaller depth, which are used for computing
    """
    
    def __init__(self, depth=0, num_variables=1):
        super(NestedFormula, self).__init__()
        self.depth = depth
        self.num_variables = num_variables
        self.subformulas = nn.ModuleList()
        # When depth is zero, formula is just a real number
        if depth == 0:
            new_lambda = nn.Parameter((2 * torch.randn((1, 1)))).requires_grad_(True)
            self.register_parameter("lambda_0", new_lambda)
            new_rational_lambda = nn.Parameter(torch.tensor([0., 0.])).requires_grad_(False)
            self.register_parameter("rational_lambda_0", new_rational_lambda)
        else:
            for i in range(self.num_variables):
                # When depth is 1, we do not need to create subformulas, since they would be just real numbers
                if self.depth != 1:
                    subformula = RecursiveFormula(self.depth - 1, self.num_variables)
                    self.subformulas.append(subformula)
                new_lambda = nn.Parameter((2 * torch.randn((1, 1)))).requires_grad_(True)
                new_power = nn.Parameter((2 * torch.randn((1, 1)))).requires_grad_(True)
                new_rational_lambda = nn.Parameter(torch.tensor([0., 0.])).requires_grad_(False)
                new_rational_power = nn.Parameter(torch.tensor([0., 0.])).requires_grad_(False)
                self.register_parameter("lambda_{}".format(i), new_lambda)
                self.register_parameter("power_{}".format(i), new_power)
                self.register_parameter("rational_lambda_{}".format(i), new_rational_lambda)
                self.register_parameter("rational_power_{}".format(i), new_rational_power)
            self.last_subformula = NestedFormula(self.depth - 1, self.num_variables)
                                    
    def forward(self, x):
        """
        Iterate over subformulas, recursively computing result using results of subformulas
        """
        # When depth is 0, we just return the corresponding number
        if self.depth == 0:
            return self.get_lambda(0).repeat(x.shape[0], 1).to(x.device)
        
        ans = torch.zeros(x.shape[0], 1).to(x.device)
        for i in range(self.num_variables):
            x_powered = torch.t(x[:, i]**self.get_power(i))
            subformula_result = torch.ones((x.shape[0], 1)).to(x.device)
            # When depth is 1, we do not need to compute subformulas
            if self.depth != 1:
                subformula_result = self.subformulas[i](x)
            ans += self.get_lambda(i) * x_powered * subformula_result           
        ans += self.last_subformula(x)
        return ans
    
    # def to(self, device):
    #     self_state_dict = self.state_dict()
    #     for key, value in self_state_dict.items():
    #         self_state_dict[key] = value.to(device)
    #     self.load_state_dict(self_state_dict)
    #     return self
    def cuda(self, device=None):
        self = super().cuda(device)        
        return self 

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)    
        return self

    def simplify(self, X_val, y_val, max_denominator=10, inplace=False):
        """
        Simplifies the formula, iterating over all its parameters and trying to substitute them with close rational number
        
        Parameters:
            X_val: torch.tensor, shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and n_features is the number of features.
            y_val: torch.tensor, shape (n_samples, 1)
                Target vector relative to X.
            max_denominator: int
                algorithm tries rational numbers with denominator not greater than max_denominator
            inplace: bool
                if True, when modify the original formula
                otherwise return new formula, leaving original one unchanged
        Returns:
            self, if inplace set to True
            simplified_version otherwise
        """
        
        simplified_version = copy.deepcopy(self)  
        simplified_state_dict = simplified_version.state_dict()
        
        # Iterate over all parameters
        for key, value in self.state_dict().items():
            if "rational" not in key: # We do not simplify rational parameters - they will be the result of simplification
                simplified_version_for_iteration = copy.deepcopy(simplified_version)
                simplified_state_dict_for_iteration = simplified_version_for_iteration.state_dict()
                y_predict = simplified_version(X_val)
                loss = nn.MSELoss()(y_val, y_predict)
                descriptive_length_of_loss = descriptive_length_of_real_number(loss)
                descriptive_length_of_existing_parameter = descriptive_length_of_real_number(value)

                # Iterate over all possible denominators
                for possible_denominator in range(1, max_denominator + 1):
#                     print("trying denominator", possible_denominator)
                    simplified_parameter_numerator = torch.round(value * possible_denominator)
                    simplified_state_dict_for_iteration[key] = simplified_parameter_numerator / possible_denominator
                    simplified_version_for_iteration.load_state_dict(simplified_state_dict_for_iteration)
                    descriptive_length_of_simplified_parameter = descriptive_length_of_fraction(simplified_parameter_numerator, possible_denominator)
#                     print(simplified_parameter_numerator, possible_denominator)
                    y_predict_simplified = simplified_version_for_iteration(X_val)
                    loss_of_simplified_model = nn.MSELoss()(y_val, y_predict_simplified)
                    descriptive_length_of_loss_of_simplified_model = descriptive_length_of_real_number(loss_of_simplified_model)                
                    # If the descriptive length did not improve, revert the change.
#                     print("descriptive_length_of_loss_of_simplified_model", descriptive_length_of_loss_of_simplified_model)
#                     print("descriptive_length_of_simplified_parameter", descriptive_length_of_simplified_parameter)
#                     print("descriptive_length_of_loss", descriptive_length_of_loss)
#                     print("descriptive_length_of_existing_parameter", descriptive_length_of_existing_parameter)

                    if descriptive_length_of_loss_of_simplified_model + descriptive_length_of_simplified_parameter > descriptive_length_of_loss + descriptive_length_of_existing_parameter:
                        simplified_version_for_iteration.load_state_dict(simplified_state_dict)
                    else:
                        # If we are successful, we update everything
                        simplified_state_dict[AddRationalInName(key)] = torch.tensor([simplified_parameter_numerator, possible_denominator])
                        simplified_version.load_state_dict(simplified_state_dict)
                        simplified_version_for_iteration = copy.deepcopy(simplified_version)
                        simplified_state_dict_for_iteration = simplified_version_for_iteration.state_dict()

                simplified_state_dict = simplified_state_dict_for_iteration
                simplified_version.load_state_dict(simplified_state_dict)
        
        if inplace:
            self = copy.deepcopy(simplified_version)
        else:
            return simplified_version      
    
    def get_lambda(self, i):
        return self.__getattr__('lambda_{}'.format(i))
    
    def get_rational_lambda(self, i):
        return self.__getattr__('rational_lambda_{}'.format(i))
    
    def get_power(self, i):
        return self.__getattr__('power_{}'.format(i))
    
    def get_rational_power(self, i):
        return self.__getattr__('rational_power_{}'.format(i))
    
    def __repr__(self):
        """
        Return tex-style string, recursively combining result from representation of subformulas
        """
        if self.depth == 0:
            if self.get_rational_lambda(0)[1] > 0: # if it is equal to 0, it means that there is no rational value
                return FormFractionRepresentation(self.get_rational_lambda(0))            
            return FormReal(self.get_lambda(0))
        
        ans = ["\left("]
        for i in range(self.num_variables):
            # First we add lambda
            if i != 0 and self.get_lambda(i) > 0:
                ans.append(" + ")
            if self.get_rational_lambda(i)[1] > 0:
                ans.append(FormFractionRepresentation(self.get_rational_lambda(i)))
            else:
                ans.append(FormReal(self.get_lambda(i)))   
            # Then we add variable and its power
            ans.append("x_{}^".format(i + 1) + "{")
            if self.get_rational_power(i)[1] > 0:
                ans.append(FormFractionRepresentation(self.get_rational_power(i)))
            else:
                ans.append(FormReal(self.get_power(i)))  
            ans += "}"    
            # Then we add the corresponding subformula
            if self.depth != 1:
                ans.append(str(self.subformulas[i]))
        if self.last_subformula.get_lambda(0) > 0:        
            ans.append(" + ")
        ans.append(str(self.last_subformula))
        ans.append(r"\right)")
        ans = ''.join(ans)
        return ans
    
    
def LearnFormula(X, y, optimizer_for_formula=torch.optim.Adam, device=torch.device("cpu"), n_init=10, max_iter=10000, 
             lr=0.01,
             depth=1, verbose=2, verbose_frequency=5000, 
             max_epochs_without_improvement=1000,
             minimal_acceptable_improvement=1e-6, max_tol=1e-5, use_swa=False):
    """
    Parameters:
        X: torch.tensor, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.
        y: torch.tensor, shape (n_samples, 1)
            Target vector relative to X.
        n_init: int 
            number of times algorithm will be run with different initial weights. 
            The final results will be the best output of n_init consecutive runs in terms of loss.
        max_iter: int 
            Maximum number of iterations of the algorithm for a single run.
        depth: int
            depth of formula to learn
        verbose: int
            if is equal to 0, no output
            if is equal to 1, output number of runs and losses
            if is equal to 2, output number of runs and losses and print loss every verbose_frequency epochs
        verbose_frequency: int
            if verbose equals 2, then print loss every verbose_frequency epochs
        max_epochs_without_improvement: int
            if during this number of epochs loss does not decrease more than minimal_acceptable_improvement, the learning process
            will be finished
        minimal_acceptable_improvement: float
            if during max_epochs_without_improvement number of epochs loss does not decrease more than this number, 
            the learning process will be finished
        max_tol: float
        	if the loss becomes smaller than this value, stop performing initializations and finish the learning process
            
    Returns:
        best_formula: RecursiveFormula
            fitted formula
        best_losses: list of float
        	loss values for best initialization
    """
    
    best_formula = NestedFormula(depth, X.shape[1]).to(device)
    best_loss = 1e20
    best_losses = []
    
    for init in range(n_init):
        losses = []
        if verbose > 0:
            print("  Initialization #{}".format(init + 1))
    #     torch.random.manual_seed(seed)
        model = NestedFormula(depth, X.shape[1]).to(device)
        
        criterion = nn.MSELoss()
        epochs_without_improvement = 0
        epoch = 0
        output = model(X)
        previous_loss = criterion(output, y).item() 
        
        if use_swa:
          	base_optimizer = torch.optim.SGD(model.parameters(), lr)        
          	optimizer = torchcontrib.optim.SWA(base_optimizer, swa_start=10, swa_freq=5, swa_lr=0.05)
        else:
          	optimizer = optimizer_for_formula(model.parameters(), lr)
        optimizer.zero_grad()

        while epoch < max_iter and epochs_without_improvement < max_epochs_without_improvement:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            losses.append(loss.item())
            loss.backward()
            if verbose == 2 and (epoch + 1) % verbose_frequency == 0:
                print("    Epoch {}, current loss {:.3}, current formula ".format(epoch + 1, loss.item()), end='')
                PrintFormula(model, "fast")       
            optimizer.step()  
            epoch += 1
            if torch.abs(previous_loss - loss) < minimal_acceptable_improvement:
                epochs_without_improvement += 1
            else:
                epochs_without_improvement = 0
            previous_loss = loss.item()
            if epoch == 1000 and loss > 1e5:
                print("  The model does not seem to converge, finishing at epoch 1000")
                epoch = max_iter
        if loss < best_loss:
            best_loss = loss
            best_formula = model
            best_losses = losses
        if verbose > 0:
            print("  Finished run #{}, loss {}, best loss {}".format(init + 1, loss, best_loss))
        if use_swa:
	        optimizer.swap_swa_sgd()
        if loss < max_tol:
            print(f'loss is smaller than {max_tol}, terminating learning process')
            break
        
    return best_formula, best_losses