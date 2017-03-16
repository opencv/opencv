import 'nn'

function fill_net(net)
	if net.modules then
		for i = 1, #net.modules do
			fill_net(net.modules[i])
		end
	end
	if net.weight then
		net.weight = torch.rand(net.weight:size())
	end
	if net.bias then
		net.bias = torch.rand(net.bias:size())
	end
	if net.train then
		net.train = 0
	end
end

function save(net, input, label)
	fill_net(net)
	output = net:forward(input)

	--torch.save(label .. '_net.dat', net)
	torch.save(label .. '_net.txt', net, 'ascii')
	--torch.save(label .. '_input.dat', input)
	torch.save(label .. '_input.txt', input, 'ascii')
	--torch.save(label .. '_output.dat', output)
	torch.save(label .. '_output.txt', output, 'ascii')

	return net
end

local net_simple = nn.Sequential()
net_simple:add(nn.ReLU())
net_simple:add(nn.SpatialConvolution(3,64, 11,7, 3,4, 3,2))
net_simple:add(nn.SpatialMaxPooling(4,5, 3,2, 1,2))
net_simple:add(nn.Sigmoid())
save(net_simple, torch.Tensor(2, 3, 25, 35), 'net_simple')

local net_pool_max = nn.Sequential()
net_pool_max:add(nn.SpatialMaxPooling(4,5, 3,2, 1,2):ceil()) --TODO: add ceil and floor modes
local net = save(net_pool_max, torch.rand(2, 3, 50, 30), 'net_pool_max')
torch.save('net_pool_max_output_2.txt', net.modules[1].indices - 1, 'ascii')

local net_pool_ave = nn.Sequential()
net_pool_ave:add(nn.SpatialAveragePooling(4,5, 2,1, 1,2))
save(net_pool_ave, torch.rand(2, 3, 50, 30), 'net_pool_ave')

local net_conv = nn.Sequential()
net_conv:add(nn.SpatialConvolution(3,64, 11,7, 3,4, 3,2))
save(net_conv, torch.rand(1, 3, 50, 60), 'net_conv')

local net_reshape = nn.Sequential()
net_reshape:add(nn.Reshape(5, 4, 3, 2))
save(net_reshape, torch.rand(2, 3, 4, 5), 'net_reshape')

local net_reshape_batch = nn.Sequential()
net_reshape_batch:add(nn.Reshape(5, 4, 3, true))
save(net_reshape_batch, torch.rand(2, 3, 4, 5), 'net_reshape_batch')

save(nn.Linear(7, 3), torch.rand(13, 7), 'net_linear_2d')

local net_parallel = nn.Parallel(4, 2)
net_parallel:add(nn.Sigmoid())
net_parallel:add(nn.Tanh())
save(net_parallel, torch.rand(2, 6, 4, 2), 'net_parallel')

local net_concat = nn.Concat(2)
net_concat:add(nn.ReLU())
net_concat:add(nn.Tanh())
net_concat:add(nn.Sigmoid())
save(net_concat, torch.rand(2, 6, 4, 3) - 0.5, 'net_concat')

local net_deconv = nn.Sequential()
net_deconv:add(nn.SpatialFullConvolution(3, 9, 4, 5, 1, 2, 0, 1, 0, 1))
save(net_deconv, torch.rand(2, 3, 4, 3) - 0.5, 'net_deconv')

local net_batch_norm = nn.Sequential()
net_batch_norm:add(nn.SpatialBatchNormalization(4, 1e-3))
save(net_batch_norm, torch.rand(1, 4, 5, 6) - 0.5, 'net_batch_norm')

local net_prelu = nn.Sequential()
net_prelu:add(nn.PReLU(5))
save(net_prelu, torch.rand(1, 5, 40, 50) - 0.5, 'net_prelu')

local net_cadd_table = nn.Sequential()
local sum = nn.ConcatTable()
sum:add(nn.Identity()):add(nn.Identity())
net_cadd_table:add(sum):add(nn.CAddTable())
save(net_cadd_table, torch.rand(1, 5, 40, 50) - 0.5, 'net_cadd_table')