import 'nn'
import 'dpnn'

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
net_pool_max:add(nn.SpatialMaxPooling(4,5, 3,2, 1,2):ceil())
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

local net_reshape_single_sample = nn.Sequential()
net_reshape_single_sample:add(nn.Reshape(3 * 4 * 5))
net_reshape_single_sample:add(nn.Linear(3 * 4 * 5, 10))
save(net_reshape_single_sample, torch.rand(1, 3, 4, 5), 'net_reshape_single_sample')

save(nn.Linear(7, 3), torch.rand(13, 7), 'net_linear_2d')

local net_reshape_channels = nn.Sequential()
net_reshape_channels:add(nn.Reshape(20))
save(net_reshape_channels, torch.rand(2, 1, 10, 2), 'net_reshape_channels')

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

local net_softmax = nn.Sequential()
net_softmax:add(nn.SoftMax())
save(net_softmax, torch.rand(1, 5, 1, 1), 'net_softmax')
save(net_softmax, torch.rand(2, 5, 3, 4), 'net_softmax_spatial')

local net_logsoftmax = nn.Sequential()
net_logsoftmax:add(nn.LogSoftMax())
save(net_logsoftmax, torch.rand(3, 4, 1, 1), 'net_logsoftmax')
save(net_logsoftmax, torch.rand(1, 6, 4, 3), 'net_logsoftmax_spatial')

local net_lp_pooling_square = nn.Sequential()
net_lp_pooling_square:add(nn.SpatialLPPooling(-1, 2, 2,2, 2,2))  -- The first argument isn't used
net_lp_pooling_square:add(nn.Tanh())
save(net_lp_pooling_square, torch.rand(3, 7, 8, 10), 'net_lp_pooling_square')

local net_lp_pooling_power = nn.Sequential()
net_lp_pooling_power:add(nn.SpatialLPPooling(-1, 3, 3,3, 2,2))  -- The first argument isn't used
net_lp_pooling_power:add(nn.Sigmoid())
save(net_lp_pooling_power, torch.rand(3, 7, 6, 7), 'net_lp_pooling_power')

local net_conv_gemm_lrn = nn.Sequential()
net_conv_gemm_lrn:add(nn.SpatialConvolutionMM(4,7, 3,3, 1,1, 1,1))
net_conv_gemm_lrn:add(nn.SpatialCrossMapLRN(3))
save(net_conv_gemm_lrn, torch.rand(2, 4, 5, 6), 'net_conv_gemm_lrn')

local net_depth_concat = nn.DepthConcat(1);
net_depth_concat:add(nn.SpatialConvolutionMM(3, 4, 1, 1))
net_depth_concat:add(nn.SpatialConvolutionMM(3, 5, 3, 3))
net_depth_concat:add(nn.SpatialConvolutionMM(3, 2, 4, 4))
save(net_depth_concat, torch.rand(2, 3, 7, 7), 'net_depth_concat')

local net_inception_block = nn.Sequential()
net_inception_block:add(nn.Inception{
	inputSize = 3,  -- Number of input channels
	kernelSize = {3},
	kernelStride = {1},
	outputSize = {4},
	reduceSize = {4},
	pool = nn.SpatialMaxPooling(3, 3, 1, 1, 1, 1),
	transfer = nn.Tanh(),
})
save(net_inception_block, torch.rand(2, 3, 16, 16), 'net_inception_block')

local net_normalize = nn.Sequential()
net_normalize:add(nn.Normalize(2))
net_normalize:add(nn.Normalize(1, 1e-3))
net_normalize:add(nn.Normalize(2.7))
save(net_normalize, torch.rand(1, 24) * 3 - 0.5, 'net_normalize')

local net_padding = nn.Sequential()
net_padding:add(nn.Padding(2, 2, 4))
net_padding:add(nn.Padding(1, -1, 2))
net_padding:add(nn.Padding(3, -3, 3, 3))
net_padding:add(nn.SpatialZeroPadding(3, 1, 2, 0));
save(net_padding, torch.rand(2, 1, 3, 4), 'net_padding')

local net_spatial_zero_padding = nn.Sequential()
net_spatial_zero_padding:add(nn.SpatialZeroPadding(1, 0, 2, 3));
save(net_spatial_zero_padding, torch.rand(4, 2, 3), 'net_spatial_zero_padding')

-- OpenFace network.
-- require 'image'
-- torch.setdefaulttensortype('torch.FloatTensor')
-- net = torch.load('../openface_nn4.small2.v1.t7')
-- net:evaluate()
-- input = image.load('../../cv/shared/lena.png')
-- input = image.scale(input, 96, 96, 'simple'):reshape(1, 3, 96, 96)
-- output = net:forward(input):reshape(1, 128)
-- torch.save('net_openface_output.dat', output)

local net_spatial_reflection_padding = nn.Sequential()
net_spatial_reflection_padding:add(nn.SpatialReflectionPadding(5, 5, 5, 5));
save(net_spatial_reflection_padding, torch.rand(1, 3, 7, 8), 'net_spatial_reflection_padding')
