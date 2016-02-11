require 'torch'
require 'optim'
Plot = require 'itorch.Plot'
torch.setdefaulttensortype('torch.DoubleTensor')

-------------------------------------------------------------------------------
-- QUESTION 3.3

-- TODO. Load the points and plot them
data = torch.load('points.dat')
x = data.inputs
t = data.targets
-- TODO
d = 5 -- define the number of parameters, namely the dimensionality of your parameter vector
      -- Note that this is a global variable, so you can access it from everywhere
n = x:size(1) -- define the number of data points, which you want to fit, namely the number of entries in your data structure
function feval(theta)
    
    local f = function(theta)
        
        local xpow = torch.pow(x,2)
        local t1 = torch.mul(xpow,theta[2])
        local power = torch.mul(torch.abs(x),theta[4])-torch.mul(xpow,theta[5])
        local texp = torch.mul(torch.exp(power),theta[3])

        local fval = torch.add(t1,texp)
        fval = torch.add(fval,theta[1])
        --Compute the mean squared error
        local mse = torch.pow(fval-t,2)
        mse = torch.sum(mse)
        

        -- Compute the loss function .we "hypothetically use 1/2 for convenience and exclude it from the gradient function below and average.
         local loss = mse/n

        return loss, fval
    end

    local g = function(theta, fval)
        -- here we store the partial derivatives
        local fgradient = torch.zeros(n,d)
        
        --x^2 local term
        local xpow = torch.pow(x,2)
        -- theta3|X| - theta4x^2 term 
        local power = torch.mul(torch.abs(x),theta[4])-torch.mul(xpow,theta[5])
        --theta0 partial
        fgradient[{{},1}] = 1
        --theta1 partial
        fgradient[{{},2}] = xpow
        --theta2 partial
        fgradient[{{},3}] = torch.exp(power)
        local g1  = torch.mul(torch.exp(power),theta[3])
        fgradient[{{},4}] = torch.cmul(torch.abs(x),g1)
        fgradient[{{},5}] = torch.cmul(torch.mul(-xpow,theta[3]),g1)
        local loss = torch.mul(fval - t, 2):resize(n,1)
        loss = torch.expand(loss,100,5)
        
        -- pass to the partials the loss function term
        fgradient:cmul(loss)
        local grad = torch.sum(fgradient,1)
        grad = torch.div(grad,n)
    	return grad
	end
 
	-- Above we defined the functions locally.
	-- We could have defined them in another file for that matter, but this way is perhaps more compact and less cluttered.
	-- In any case suit yourself, pick your way.
	-- Here, you call the functions that you have implemented before.
	-- Their output is what feval should return in the end.
    local loss, fval = f(theta)
    local z = fval:clone()
    local grad = g(theta, z)
    
    return loss, grad, fval
end

-------------------------------------------------------------------------------
-- QUESTION 3.4

-- TODO define your state variable
--Use verbose true for more output
state = {
   learningRate = 0.01,
   maxIter = 1000,
   tol = 0.01,
   verbose = false
}

-- TODO define your initialization.
-- You can pick a random vector using torch.randn
-- Naturally, the parameters theta as well as the gradient wrt to the parameters should have the same dimensionality

function optimizeFunction(flag)
    local flag = flag or false
    local theta = torch.rand(5)
    local grad = torch.Tensor(d)

    -- TODO fill in the code for the optimization and run it
    -- stop when the gradient is close to 0, or after many iterations
    local lossall = torch.Tensor(state.maxIter+1):zero()
    local iter = 1
    while true do
         -- Using cg  optimization function.
        if flag == true then
            theta, f_ = optim.cg(feval, theta, state)
        else
            theta, f_ = optim.sgd(feval, theta, state)
        end
        lossall[iter] = f_[1]
        _,grad,_ = feval(theta)
        -- gradient norm is SOMETIMES a good measure of how close we are to the optimum, but often not.
        -- the issue is that we'd stop at points like x=0 for x^3
        gnorm = torch.norm(grad)
        if(state.verbose == true) then
            io.write('||grad||_2:',gnorm,' iteration:',iter,' error:',lossall[iter]/n*100,'\n')
        end
        if gnorm < state.tol or iter > state.maxIter then 
            break 
        end
        iter = iter + 1
    end

    io.write('Max iteration:',iter,'\n')
    io.write('f optimum:',f_[1],'\n')
    io.write('Error at last iteration:',lossall[iter]/n*100,'\n')
    io.write('Optimal parameters:')
    local i = 0
    theta:apply(function()
        i = i+1
        io.write(theta[i],'\n')
        return
    end)

    -- Run the feval once more if you want to get the outputs based on the final, optimal theta
    local loss, grad, z = feval(theta)

    local sx,ind = torch.sort(x)
    local sz = torch.Tensor(n)
    for i=1, sx:size(1) do
        sz[i] = z[ind[i]]
    end

    -- Plot the curve and the points. Do they make sense?
    Plot():line(sx, sz, 'red'):circle(x,z,'green'):circle(x, t, 'blue'):title('Data points'):draw()
    Plot():line(torch.range(1, iter), lossall[{{1,iter}}], 'green'):title('Training loss'):draw()
    return
end
print('CG optimization')
optimizeFunction(true)
print('SGD optimization')
optimizeFunction(false)