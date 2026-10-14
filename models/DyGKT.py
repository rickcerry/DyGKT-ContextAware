import numpy as np
import torch
import torch.nn as nn
from utils.utils import NeighborSampler
from models.modules import TimeEncoder, TimeDualDecayEncoder

# neighbor sampler strategy: recent

class DyGKT(nn.Module):
    def __init__(self,node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 time_dim = 16,
                 num_neighbors : int = 50,
                 ablation = '-1',
                 contextual_sampling: bool = False,
                 dropout : float = 0.5,
                 device:str='cuda:0'):

        super(DyGKT, self).__init__()

        self.num_neighbors = num_neighbors
        self.ablation = ablation
        self.contextual_sampling = contextual_sampling

        self.num_skills = int(np.unique(node_raw_features[:, 0]).max()) + 1
        self.num_nodes = node_raw_features.shape[0]

        self.node_raw_features = torch.from_numpy(
            node_raw_features.astype(np.float32)
        ).to(device)

        self.edge_raw_features = torch.from_numpy(
            edge_raw_features.astype(np.float32)
        ).to(device)

        # Kept on CPU because contextual neighbor selection is performed by the
        # NumPy-based sampler before embeddings are moved to the model device.
        self.node_skills = np.asarray(node_raw_features[:, 0])

        self.edge_dim = 64
        self.node_dim = 64
        self.time_dim = time_dim

        self.projection_layer = nn.ModuleDict({
            'feature_Linear':nn.Linear(in_features=self.node_raw_features.shape[-1], out_features=self.node_dim, bias=True),
            # 'feature_Embed':nn.Embedding(self.num_skills, self.node_dim),
            # 'node': nn.Embedding(self.num_nodes, self.node_dim),
            'edge': nn.Linear(in_features=1, out_features=self.node_dim, bias=True),
            'time': nn.Linear(in_features=self.time_dim, out_features=self.node_dim, bias=True),
            'struct': nn.Linear(in_features=1, out_features=self.node_dim, bias=True),
        })

        self.output_layer = nn.Linear(in_features=self.node_dim, out_features=self.node_dim, bias=True)

        self.dropout = dropout
        self.device = device

        
        self.src_node_updater = DyKT_Seq(edge_dim=self.edge_dim, node_dim=self.node_dim)
        self.dst_node_updater = DyKT_Seq(edge_dim=self.edge_dim, node_dim=self.node_dim)
        if self.ablation == 'dual':
            self.time_encoder = TimeEncoder(time_dim=self.time_dim)
        else:
            self.time_encoder = TimeDualDecayEncoder(time_dim=self.time_dim)# TimeEncoder(time_dim=self.time_dim)
        
    def set_neighbor_sampler(self,neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler

    def compute_src_dst_node_temporal_embeddings(self,src_node_ids: np.ndarray, edge_ids:np.ndarray,
                                                node_interact_times: np.ndarray,dst_node_ids: np.ndarray):
        #    def get_historical_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        if self.contextual_sampling:
            target_skills = self.node_skills[dst_node_ids]

            # Student history is re-ranked using the skill of the current question.
            src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = \
                self.neighbor_sampler.get_neighbors_dst(
                    node_ids=src_node_ids,
                    target_skill=target_skills,
                    node_interact_times=node_interact_times,
                    edge_ids=edge_ids
                )

            # Question history remains purely recency-based.
            dst_neighbor_node_ids, dst_neighbor_edge_ids, dst_neighbor_times = \
                self.neighbor_sampler.get_neighbors_src(
                    node_ids=dst_node_ids,
                    node_interact_times=node_interact_times,
                    edge_ids=edge_ids
                )
        else:
            src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = \
                self.neighbor_sampler.get_historical_neighbors(
                    src_node_ids, node_interact_times, self.num_neighbors
                )
            dst_neighbor_node_ids, dst_neighbor_edge_ids, dst_neighbor_times = \
                self.neighbor_sampler.get_historical_neighbors(
                    dst_node_ids, node_interact_times, self.num_neighbors
                )

        # Normalize the legacy NumPy sampler and the contextual CUDA sampler to
        # device tensors. From here onward the embedding path stays on-device.
        src_neighbor_node_ids = torch.as_tensor(src_neighbor_node_ids, device=self.device)
        src_neighbor_edge_ids = torch.as_tensor(src_neighbor_edge_ids, device=self.device)
        src_neighbor_times = torch.as_tensor(src_neighbor_times, dtype=torch.float32, device=self.device)
        dst_neighbor_node_ids = torch.as_tensor(dst_neighbor_node_ids, device=self.device)
        dst_neighbor_edge_ids = torch.as_tensor(dst_neighbor_edge_ids, device=self.device)
        dst_neighbor_times = torch.as_tensor(dst_neighbor_times, dtype=torch.float32, device=self.device)
        src_ids_t = torch.as_tensor(src_node_ids, device=self.device)
        dst_ids_t = torch.as_tensor(dst_node_ids, device=self.device)
        interact_times_t = torch.as_tensor(node_interact_times, dtype=torch.float32, device=self.device)

        src_neighbor_node_ids = torch.cat((src_neighbor_node_ids, src_ids_t[:, None]), dim=1)
        src_neighbor_edge_ids = torch.cat((src_neighbor_edge_ids, torch.zeros_like(src_ids_t[:, None])), dim=1)
        src_neighbor_times = torch.cat((src_neighbor_times, interact_times_t[:, None]), dim=1)
        dst_neighbor_node_ids = torch.cat((dst_neighbor_node_ids, dst_ids_t[:, None]), dim=1)
        dst_neighbor_edge_ids = torch.cat((dst_neighbor_edge_ids, torch.zeros_like(dst_ids_t[:, None])), dim=1)
        dst_neighbor_times = torch.cat((dst_neighbor_times, interact_times_t[:, None]), dim=1)

        # Transfer each index array once. All comparisons and feature gathers
        # below then stay on the GPU instead of triggering implicit transfers.
        src_neighbor_ids_t = src_neighbor_node_ids
        dst_neighbor_ids_t = dst_neighbor_node_ids

        src_nodes_neighbor_co_occurrence_features = (
            src_neighbor_ids_t[:, :-1] == dst_ids_t.unsqueeze(1)
        ).unsqueeze(-1).float()
        dst_nodes_neighbor_co_occurrence_features = (
            dst_neighbor_ids_t[:, :-1] == src_ids_t.unsqueeze(1)
        ).unsqueeze(-1).float()

        src_node_skill = self.node_raw_features[src_neighbor_ids_t][:, :-1, 0].long()
        dst_node_skill = self.node_raw_features[dst_neighbor_ids_t][:, -1, 0].long().unsqueeze(1)

        src_nodes_neighbor_skill_features = (src_node_skill == dst_node_skill).unsqueeze(-1).float()
        a = 1
        if self.ablation == 'counter':
            a = 0
        
        src_nodes_neighbor_struct_features = self.projection_layer['struct'](a * src_nodes_neighbor_co_occurrence_features)
        dst_nodes_neighbor_struct_features = self.projection_layer['struct'](a * dst_nodes_neighbor_co_occurrence_features)
        src_nodes_neighbor_skill_struct_features = self.projection_layer['struct'](a * src_nodes_neighbor_skill_features)

        src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features = self.get_features(
            node_interact_times=interact_times_t, nodes_edge_ids=src_neighbor_edge_ids,
            nodes_neighbor_ids=src_neighbor_node_ids, nodes_neighbor_times=src_neighbor_times)
        dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features = self.get_features(
            node_interact_times=interact_times_t, nodes_edge_ids=dst_neighbor_edge_ids,
            nodes_neighbor_ids=dst_neighbor_node_ids, nodes_neighbor_times=dst_neighbor_times)
        
        src_nodes_features = src_nodes_neighbor_node_raw_features + src_nodes_edge_raw_features + src_nodes_neighbor_time_features #+ src_nodes_neighbor_struct_features  # torch.cat((src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features),dim=-1) 
        dst_nodes_features = dst_nodes_neighbor_node_raw_features + dst_nodes_edge_raw_features + dst_nodes_neighbor_time_features #+ dst_nodes_neighbor_struct_features  # torch.cat((dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features), dim=-1) 

        src_node_embeddings = self.src_node_updater.update(
            src_nodes_features[:, :-1, :] + src_nodes_neighbor_skill_struct_features+ src_nodes_neighbor_struct_features) + (src_nodes_edge_raw_features + src_nodes_neighbor_time_features)[:,-1, :]
        
        if self.ablation in ['q_qid', 'q_kid']:
            dst_node_embeddings = dst_nodes_neighbor_node_raw_features[:,-1]
        else:
            dst_node_embeddings = self.dst_node_updater.update((dst_nodes_edge_raw_features + dst_nodes_neighbor_time_features)[:, :-1, :]+ dst_nodes_neighbor_struct_features) + dst_nodes_features[:,-1,:] 


        

        src_node_embeddings = self.output_layer(src_node_embeddings)
        dst_node_embeddings = self.output_layer(dst_node_embeddings)
        
        return src_node_embeddings, dst_node_embeddings


    def get_features(self, nodes_neighbor_ids: np.ndarray, nodes_edge_ids: np.ndarray,
                     nodes_neighbor_times: np.ndarray,node_interact_times: np.ndarray):
        node_ids_t = torch.as_tensor(nodes_neighbor_ids, device=self.device)
        edge_ids_t = torch.as_tensor(nodes_edge_ids, device=self.device)

        if self.ablation in ['embed', 'q_kid']:
            nodes_neighbor_node_raw_features = self.projection_layer['feature_Embed'](self.node_raw_features[node_ids_t][:,:,0].long())
        elif self.ablation == 'q_qid':
            nodes_neighbor_node_raw_features = self.projection_layer['node'](node_ids_t)
        else:
            nodes_neighbor_node_raw_features = self.projection_layer['feature_Linear'](self.node_raw_features[node_ids_t])
        
        if self.ablation == 'dual':
            time_values = node_interact_times[:, np.newaxis] - nodes_neighbor_times
        else:   
            time_values = nodes_neighbor_times

        nodes_neighbor_time_features = self.time_encoder(
            torch.as_tensor(time_values, dtype=torch.float32, device=self.device)
        )
        
        nodes_neighbor_time_features = self.projection_layer['time'](nodes_neighbor_time_features)
        nodes_edge_raw_features = self.projection_layer['edge'](
            self.edge_raw_features[edge_ids_t][:, :, 0].unsqueeze(-1)
        )
        
        if self.ablation == 'time':
            nodes_neighbor_time_features *= 0
        elif self.ablation == 'skill':
            nodes_neighbor_node_raw_features *= 0
        
        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features
    
class DyKT_Seq(nn.Module):

    def __init__(self, edge_dim : int,node_dim: int):
        """
        GRU-based memory updater.
        :param memory_bank: MemoryBank
        :param message_dim: int, dimension of node messages
        :param memory_dim: int, dimension of node memories
        """
        super(DyKT_Seq,self).__init__()
        self.patch_enc_layer = nn.Linear(edge_dim, node_dim)

        self.hid_node_updater = nn.GRU(input_size=edge_dim, hidden_size=node_dim,batch_first=True)# LSTM


    def update(self, x):
        outputs, hidden = self.hid_node_updater(x)

        return torch.squeeze(hidden,dim=0)

