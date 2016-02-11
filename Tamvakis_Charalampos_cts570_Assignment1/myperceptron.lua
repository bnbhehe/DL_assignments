
require 'optim'
require 'torch'
Plot = require 'itorch.Plot'



--Sample data from 2-D gaussians with μ,σ
function sampleData(mu1,sigma1,mu2,sigma2,N)

    local mu1 = mu1 or -3
    local sigma1 = sigma1 or 1
    
    local mu2 = mu2 or 3
    local sigma2 = sigma2 or 1
    local N = N or 500
    -- Random samples from ~N(mu1,sigma1)
    local randoms = torch.add(torch.mul(torch.randn(N,2),sigma1),mu1)
    local randoms = torch.cat(randoms,torch.ones(N,1))
    -- Random samples from ~N(mu2,sigma2)
    local randoms2 = torch.add(torch.mul(torch.randn(N,2),sigma2),mu2)
    local randoms2 = torch.cat(randoms2,torch.mul(torch.ones(N,1),-1))
    local data = torch.cat(randoms,randoms2,1)
    Plot():gscatter(data[{{},1}],data[{{},2}],data[{{},3}]):title('Training 2D samples'):draw()
    
    -- We don' permutate the data because either way we are using batch gradient descent.
return data
    
end

function perceptron_train(data,state)

    local eta = state.learningRate or 0.1
    local maxIter = state.maxIter or 100
    local verbose = state.verbose or false
    local iter = 1
    local error_rate = torch.zeros(state.maxIter+1)
    local N = data:size(1)
    -- Our data are the 1st 2nd dimensions
    local X = data[{{},{1,2}}]
    -- Concatenating another x0=1 column as bias
    local X = torch.cat(torch.ones(N,1),X)
    -- Real values
    local Ttrue = data[{{},{3}}]
    local W = torch.zeros(1,3)
    
    while true do
        
        Y = torch.sign(X*W:t())
        -- Elements which are 0 means are of the same class
        local sub = torch.csub(Ttrue,Y)
        -- sub:ne(0) returns a vector of 1s in the indexes where sub(index) != 0.
        -- The sum is the correctly classified.
        classified = torch.sum(sub:ne(0))
       

        
        local Winc = torch.mul(sub:t()*X,eta)
        -- Gradient descent
        W = W+Winc
        error_rate[iter] = (classified/N)*100
        if verbose == true then
            io.write('Iteration:',iter,' misclassified:',N-classified,' Error rate:',error_rate[iter],'%\n')
        end
        if (iter > maxIter-1 or classified == 0) then
            break
        end
        iter = iter + 1
    end
    
    io.write('Finished at iteration:',iter,'\n')
    io.write('Learning Rate used:',eta,'\n')
    io.write('Misclassified:',classified,'\n')
    io.write('Error at max iteration:',error_rate[iter],'%\n')
    --Save the model and return it for plotting boundary.
    model = {
        traindata = data,
        weights = W,
        _error = error_rate
    }
    --Also plot error rate line if requested
    if verbose == true then
        Plot():line(torch.range(1,iter),error_rate[{{1,iter}}]):title('Error Rate'):draw()
    end
    return model
end

function plot_boundary(model)
    
    W = model.weights:t()
    data = model.traindata
    
    
    local minX = torch.min(torch.min(data[{{},{1,2}}],1),2)
    local maxX = torch.max(torch.max(data[{{},{1,2}}],1),2)

    
    y1 = ( W[2][1]*minX[1][1] + W[1][1]) / -W[3][1];

    y2 = ( W[2][1]*maxX[1][1] + W[1][1]) / -W[3][1];
    x = torch.Tensor({minX[1][1],maxX[1][1]})
    y = torch.Tensor({y1,y2})
   
    Plot():line(x,y,'black','Boundary'):title('Decision Boundary plot')
    :gscatter(data[{{},1}],data[{{},2}],data[{{},3}]):draw()
    return true
end




-- Initialize state of training parameters.
state = {
    learningRate = 0.01,
    maxIter = 150,
    --for error printing
    verbose = true
}

-- First scenario with ~N(-3,1) and N(3,1) with 500 data points
print('First scenario. Linearly separable set.')
data = sampleData(-3,1,3,1,500)
model = perceptron_train(data,state)
plot_boundary(model)

print('Second scenario. Non linearly separable. Gaussian clusters are overlapping.Outputs provided below')
data = sampleData(-0.5,1,0.5,1,500)
state.verbose = true
model = perceptron_train(data,state)
plot_boundary(model)


