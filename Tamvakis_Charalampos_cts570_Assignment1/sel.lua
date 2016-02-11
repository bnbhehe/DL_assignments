require 'torch'
require 'optim'
Plot = require 'itorch.Plot'
torch.setdefaulttensortype('torch.DoubleTensor')

-------------------------------------------------------------------------------
-- QUESTION 3.3

-- Load the points and plot them
data = torch.load('points.dat')
x = data.inputs
t = data.targets

-- Define dimensions
d = 5
n_help = x:size()
n = n_help[1]

function feval(theta)
    
    local f = function(theta)

		local term_1 = theta[1] 
		local term_2 = torch.mul(torch.pow(x,2), theta[2])
		local term_3_subterm_1 = torch.mul(torch.abs(x), theta[4])
		local term_3_subterm_2 = torch.mul(torch.pow(x,2), theta[5])
		local term_3_helper = torch.exp(term_3_subterm_1 - term_3_subterm_2)
		local term_3 = torch.mul(term_3_helper, theta[3])
		-- t1 + (t2*x^2) + (t3*(exp((t4*abs(x))-(t5*x^2))))
		local fval_helper = torch.add(term_2, term_1)
        local fval = torch.add(fval_helper, term_3)

		local squares = torch.pow(fval - t,2)
		local sm = torch.sum(squares)
		local tot = sm / n
        local loss = tot

        return loss, fval
    end

    local g = function(theta, fval)
	local fgradient = torch.Tensor(n,d):fill(0)

	fgradient[{{},1}] = torch.Tensor(n):fill(1)	

	fgradient[{{},2}] = torch.pow(x,2)

		local fgradient_theta_3_subterm_1 = torch.mul(torch.abs(x), theta[4])
		local fgradient_theta_3_subterm_2 = torch.mul(torch.pow(x,2), theta[5])
	fgradient[{{},3}] = torch.exp(fgradient_theta_3_subterm_1 - fgradient_theta_3_subterm_2)

		local fgradient_theta_4_helper = torch.mul(torch.abs(x), theta[3])
	fgradient[{{},4}] = torch.cmul(fgradient_theta_4_helper, fgradient[{{},3}]:clone())

		local fgradient_theta_5_helper = torch.mul(torch.pow(x,2), theta[3])
	fgradient[{{},5}] = torch.cmul(fgradient_theta_5_helper, fgradient[{{},3}]:clone())


		local error_constant_helper = torch.mul(fval - t, 2):resize(n,1)
	local error_constant = torch.expand(error_constant_helper, 100, 5)

		local ggradient_helper = torch.Tensor(n,d):fill(0)
		ggradient_helper = torch.cmul(error_constant, fgradient)

		local grad_helper = torch.sum(ggradient_helper, 1)
	local grad = torch.div(grad_helper, n)

    	return grad
	end
 

    local loss, fval = f(theta)
	local z = fval:clone()
    local grad = g(theta, z)

    return loss, grad, fval
end

-------------------------------------------------------------------------------
-- QUESTION 3.4

state = {
   maxIter = 100
}

theta = torch.randn(d)
grad = torch.Tensor(d)


lossall = torch.Tensor(state.maxIter+1):zero()
iter = 1
while true do
    -- optim has multiple functions, such as adagrad, sgd, lbfgs, and others
    theta, f_ = optim.cg(feval, theta, state)
    lossall[iter] = f_[1]
    
    if iter > state.maxIter then 
        break 
    end
    iter = iter + 1
end

print('f optimal: ' .. f_[1])
print('Optimal parameters:')
print(theta)
print('[Targets, Predictions]:')

loss, grad, z = feval(theta)
print(torch.cat(t, z, 2))

local sorted_x, index_x  = torch.sort(x:clone())
local sorted_z = torch.Tensor(n)

for i=1, sorted_x:size()[1] do
	local assignment = index_x[i]
	sorted_z[i] = z[assignment]
end

Plot():line(sorted_x, sorted_z, 'red'):circle(x,z,'green'):circle(x, t, 'blue'):title('Data points'):draw()
Plot():line(torch.range(1, state.maxIter+1), lossall, 'green'):title('Training loss'):draw()
--
