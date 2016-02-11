-- Fibonacci function
function fib(n)
    if(n<2) then return n end
        return fib(n-2)+fib(n-1)
end
-- Custom tensor print Function.
function myprint(t)
    for i=1,t:size(1) do
        for j=1,t:size(2) do
            io.write(t[i][j],'\t')
        end
        io.write('\n')
    end
end
----------------------- QUESTION 1.1 --------------------------------
print('Question 1.1')
--Initialize tensor
a = torch.Tensor(3,5)
--Apply function to tensor . This function is to fill each cell with the fibonacci numbers row-wise
i = 1
a:apply(function()
 
  i = i + 1
  return  fib(i)
end)
io.write('Primary Tensor with fibonacci numbers row-wise:\n')
myprint(a) -- s is interpreted by a as a 2D matrix
print('')
----------------------- QUESTION 1.2 --------------------------------
--Slice tensor as shown in image

print('Question 1.2:Slice Tensor to sub tensors')
print('')

redsub = a[{1,1}]
io.write('Red sub-tensor:',redsub,'\n')
print('')
brownsub = a[{{3}, {}}]

io.write('Brown sub-tensor:')
myprint(brownsub)
print('')
greensub = a[{{}, {2}}]
io.write('Green sub-tensor:')
myprint(greensub:t())
print('')
bluesub = a[{{}, {2,4}}] or a[{2, {2,4}}]
io.write('Blue sub-tensor:\n')
myprint(bluesub)
print('')
---------------------- QUESTION 1.3 ---------------------------------
print('Question 1.3: \n')
c = a[{{1},{}}]
for i=2 , a:size(1) do
    rowvec = a[{{i},{}}]
    c = torch.cat(c,rowvec,2)
end
io.write('2-d 15x1 tensor of primary tensor:\n')
myprint(c)
d = c[{1,{}}]

io.write('1-d of size 15 tensor of primary tensor:\n')
print(d)

io.write('15x4 Tensor of the previous tensor:\n')
e = d:repeatTensor(d,4,1):t()
print(e)
print('')
--------------------- QUESTION 1.5 ---------------------------------
print('Question 1.4: \n')
a[{{3},{}}]:fill(1.5)
a[{{},{2}}]:fill(1.5)
print('New Tensor with 1.5 values:')
myprint(a)