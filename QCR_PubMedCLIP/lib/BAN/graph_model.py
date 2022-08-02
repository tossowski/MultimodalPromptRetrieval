import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

class NodeApplyModule(nn.Module):
	def __init__(self, in_feats, out_feats, activation):
		super(NodeApplyModule, self).__init__()
		self.linear = nn.Linear(in_feats, out_feats)
		self.activation = activation

	def forward(self, node):
		h = self.linear(node.data['h'])
		h = self.activation(h)
		return {'h' : h}

class GCN(nn.Module):
	def __init__(self, in_feats, out_feats, activation):
		super(GCN, self).__init__()
		self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

	def forward(self, g, feature):
		g.ndata['h'] = feature
		g.update_all(gcn_msg, gcn_reduce)
		g.apply_nodes(func=self.apply_mod)
		return g.ndata.pop('h')


class GraphModel(nn.Module):
	def __init__(self):
		super(GraphModel, self).__init__()
		self.gcn1 = GCN(300, 512, F.relu)
		self.gcn2 = GCN(512, 1024, F.relu)
	
	def forward(self, g):
		x = self.gcn1(g, g.ndata['node_feats'])
		x = self.gcn2(g, x)
		g.ndata['h'] = x
		hg = dgl.mean_nodes(g, 'h')
		return hg
