require 'torch'
require 'optim'
Plot = require 'itorch.Plot'
torch.setdefaulttensortype('torch.DoubleTensor')

-------------------------------------------------------------------------------
-- QUESTION 2.1
print('Question 2.1')

function f(x)
    --fval = x^2+ math.exp(math.abs(x)-x^2)
    local xpow = torch.pow(x,2)
   -- print(xpow)
    local power = torch.abs(x)-xpow
    fval = torch.add(xpow,torch.exp(power))
    
    return fval
end
xrange = torch.linspace(-2, 2, 51)
frange = f(xrange)
Plot():line(xrange, frange,'blue'):title('Function graph'):draw()
-------------------------------------------------------------------------------


-- QUESTION 2.2
print('Question 2.2')
function g(x)
    
       -- grad = 2*x + math.exp(math.abs(x)-x^2)*(x/math.abs(x)-2*x)
        local xpow = torch.pow(x,2)
        local power = torch.abs(x)-xpow
        local xx = torch.mul(x,2)
        local dx = torch.cdiv(x,torch.abs(x)) - xx
        grad = xx + torch.cmul(torch.exp(power),dx)
   
        
  
    return grad
end

grange = torch.zeros(51)
grange = g(xrange)
_,nanInd = grange:ne(grange):max(1)

--print(grange)
--Hacky Method for plotting.
xl = torch.zeros(2)
yl = torch.linspace(-1,1,2)
local plot = Plot()
grange[nanInd[1]] =-1
plot:line(xrange[{{1,nanInd[1]},}],grange[{{1,nanInd[1]},}]):title('Gradient graph')
grange[nanInd[1]] = 1
plot:line(xl,yl)
plot:line(xrange[{{nanInd[1],51},}],grange[{{nanInd[1],51},}])
plot:draw()
-------------------------------------------------------------------------------
print('Question 2.3')

function feval(x)
    
       -- f(x) value
       fval = f(x)
       
       -- df(x) value
       grad = g(x)

    return fval, grad
end

-- In the optim package your need to define a table with the state variable during the optimization.
-- The state variable contains various information about the current point of the optimization, like the learning rate etc.

state = {
   learningRate = 0.1,
   tolX = 0.05,
   max_iter = 10000,
   verbose = true
}

-- Define here an initialization for x, e.g. 5
x = torch.Tensor{1}

iter = 0
while true do
    -- optim has multiple functions, such as adagrad, sgd, lbfgs, and others
    -- see documentation for more details
    -- TODO. Call the optimization function here
    -- xoptim, foptim = optim.adagrad(...
     xoptim,foptim = optim.adagrad(feval,x,state)
    -- gradient norm is SOMETIMES a good measure of how close we are to the optimum, but often not.
    -- the issue is that we'd stop at points like x=0 for x^3
    
    -- stop when the gradient is close to 0, or after many iterations, e.g. 50000.
    -- You can add this in the state variable, so that everything is neat.
   -- 
    if   iter > state.max_iter then 
        break 
    end
    iter = iter + 1
end
print('x optimum ',xoptim[1],'F optimum',foptim[1][1])
print('Plotting')
Plot():line(xrange, frange, 'red'):circle(xoptim, foptim[1], 'blue'):title('Function plot and minimum point'):draw()