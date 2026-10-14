import numpy as np
import torch
from utils.DataLoader import Data


# Class that contains the hyperparameters needed for sub-sampling
class HyperParameters:
    
    def __init__(self, eps=0.6, number_of_neighbors=50, gamma=2):
        # Number of neighbors
        if number_of_neighbors <= 0:
            raise ValueError("The number of neighbors is not valid!")
        
        self.N = int(number_of_neighbors)
        
        # Gamma: how many recent candidates we consider before contextual re-ranking
        if gamma % 1 != 0 or gamma < 1:
            raise ValueError("Gamma value is not valid!")
        
        self.gamma = int(gamma)
        
        # Tolerance value
        if eps >= 1 or eps <= 0:
            raise ValueError("The tolerance parameter is not valid!")
        
        self.eps = eps
        
        # Tau_time: decay over time
        self.tau_t = self.N / -np.log(eps)
        
        # Tau_context: additional contribution to the decay over time
        if self.N == 1:
            self.tau_c = self.gamma - 1
        else:
            self.tau_c = (self.gamma * self.N - 1) / (self.N - 1) - 1


# Builder for NeighborSampler
def get_contextual_neighbor_sampler(
    data: Data,
    hp: HyperParameters,
    node_raw_features: np.ndarray,
    seed: int = None,
    device: str = 'cpu'
):
    """
    Build the adjacency list and return the contextual neighbor sampler.

    data: Data object containing src_node_ids, dst_node_ids, edge_ids, node_interact_times
    hp: HyperParameters
    node_raw_features: numpy array containing node-level information.
                       Assumption: node_raw_features[node_id][0] is the skill/concept ID
                       for question nodes.
    """

    max_node_id = int(max(data.src_node_ids.max(), data.dst_node_ids.max()))

    # adj_list[node_id] = list of (neighbor_id, edge_id, timestamp)
    adj_list = [[] for _ in range(max_node_id + 1)]

    for src_node_id, dst_node_id, edge_id, node_interact_time in zip(
        data.src_node_ids,
        data.dst_node_ids,
        data.edge_ids,
        data.node_interact_times
    ):
        src_node_id = int(src_node_id)
        dst_node_id = int(dst_node_id)
        edge_id = int(edge_id)
        node_interact_time = float(node_interact_time)

        # Undirected adjacency
        adj_list[src_node_id].append((dst_node_id, edge_id, node_interact_time))
        adj_list[dst_node_id].append((src_node_id, edge_id, node_interact_time))

    return ContextualNeighborSampler(
        adj_list=adj_list,
        hyper=hp,
        data=data,
        node_raw_features=node_raw_features,
        seed=seed,
        device=device
    )


# New neighbor sampler that contains a trade-off between context and time
class ContextualNeighborSampler:

    def __init__(
        self,
        adj_list: list,
        hyper: HyperParameters,
        data: Data,
        node_raw_features: np.ndarray,
        seed: int = None,
        device: str = 'cpu'
    ):
        """
        Neighbor sampler.

        adj_list: list of lists.
                  adj_list[node_id] contains tuples:
                  (neighbor_id, edge_id, timestamp)

        node_raw_features:
                  Assumption:
                  node_raw_features[question_node_id][0] = skill/concept ID.
        """

        self.hyper = hyper
        self.device = torch.device(device)
        # Keep the skill column contiguous so contextual scoring can retrieve all
        # candidate skills with one NumPy indexing operation.
        self.node_skills = np.asarray(node_raw_features[:, 0])

        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []

        for per_node_neighbors in adj_list:
            # Sort neighbors by timestamp
            sorted_per_node_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])

            self.nodes_neighbor_ids.append(
                np.array([x[0] for x in sorted_per_node_neighbors], dtype=np.longlong)
            )
            self.nodes_edge_ids.append(
                np.array([x[1] for x in sorted_per_node_neighbors], dtype=np.longlong)
            )
            self.nodes_neighbor_times.append(
                np.array([x[2] for x in sorted_per_node_neighbors], dtype=np.float64)
            )

        # Compact CSR representation used by the vectorized GPU sampler.
        # The arrays remain sorted by time inside every node row.
        degrees = np.fromiter(
            (len(x) for x in self.nodes_neighbor_ids),
            dtype=np.int64,
            count=len(self.nodes_neighbor_ids)
        )
        row_ptr = np.empty(len(degrees) + 1, dtype=np.int64)
        row_ptr[0] = 0
        np.cumsum(degrees, out=row_ptr[1:])

        if row_ptr[-1] > 0:
            csr_neighbor_ids = np.concatenate(self.nodes_neighbor_ids)
            csr_edge_ids = np.concatenate(self.nodes_edge_ids)
            csr_times = np.concatenate(self.nodes_neighbor_times)
        else:
            csr_neighbor_ids = np.empty(0, dtype=np.int64)
            csr_edge_ids = np.empty(0, dtype=np.int64)
            csr_times = np.empty(0, dtype=np.float64)

        # For every edge and endpoint, store the exclusive CSR cutoff.  Using
        # side='left' is essential: the current edge and every event with the
        # same timestamp are excluded, preventing temporal leakage.
        max_edge_id = int(data.edge_ids.max()) if len(data.edge_ids) else 0
        edge_src = np.zeros(max_edge_id + 1, dtype=np.int64)
        edge_dst = np.zeros(max_edge_id + 1, dtype=np.int64)
        edge_src_end = np.zeros(max_edge_id + 1, dtype=np.int64)
        edge_dst_end = np.zeros(max_edge_id + 1, dtype=np.int64)
        for src, dst, edge, timestamp in zip(
            data.src_node_ids, data.dst_node_ids, data.edge_ids,
            data.node_interact_times
        ):
            src, dst, edge = int(src), int(dst), int(edge)
            edge_src[edge], edge_dst[edge] = src, dst
            edge_src_end[edge] = row_ptr[src] + np.searchsorted(
                self.nodes_neighbor_times[src], timestamp, side='left'
            )
            edge_dst_end[edge] = row_ptr[dst] + np.searchsorted(
                self.nodes_neighbor_times[dst], timestamp, side='left'
            )

        self.csr_row_ptr = torch.as_tensor(row_ptr, device=self.device)
        self.csr_neighbor_ids = torch.as_tensor(csr_neighbor_ids, device=self.device)
        self.csr_edge_ids = torch.as_tensor(csr_edge_ids, device=self.device)
        self.csr_times = torch.as_tensor(
            csr_times, dtype=torch.float32, device=self.device
        )
        self.edge_src = torch.as_tensor(edge_src, device=self.device)
        self.edge_dst = torch.as_tensor(edge_dst, device=self.device)
        self.edge_src_end = torch.as_tensor(edge_src_end, device=self.device)
        self.edge_dst_end = torch.as_tensor(edge_dst_end, device=self.device)
        self.node_skills_t = torch.as_tensor(
            self.node_skills, device=self.device
        )

    def _history_end(self, node_ids: torch.Tensor,
                     edge_ids: torch.Tensor) -> torch.Tensor:
        """Return the strict-before-time CSR cutoff for each query."""
        is_src = self.edge_src[edge_ids] == node_ids
        # Endpoint consistency is guaranteed by the edge mappings constructed
        # from Data. Avoid torch.all() here: converting its result to a Python
        # bool would force a CUDA synchronization on every batch.
        return torch.where(
            is_src, self.edge_src_end[edge_ids], self.edge_dst_end[edge_ids]
        )

    def _recent_candidates(self, node_ids, edge_ids, count):
        """Gather the last ``count`` strict historical events in one batch."""
        node_ids = torch.as_tensor(node_ids, dtype=torch.long,
                                   device=self.device)
        edge_ids = torch.as_tensor(edge_ids, dtype=torch.long,
                                   device=self.device)
        end = self._history_end(node_ids, edge_ids)
        start = self.csr_row_ptr[node_ids]
        offsets = torch.arange(count, device=self.device)
        positions = end[:, None] - count + offsets[None, :]
        valid = positions >= start[:, None]
        positions = positions.clamp_min(0)
        return (
            self.csr_neighbor_ids[positions],
            self.csr_edge_ids[positions],
            self.csr_times[positions],
            valid
        )

    def find_neighbors_before(self, node_id: int, interact_time: float):
        """
        Return all historical neighbors of node_id before interact_time.

        Output arrays are sorted from oldest to newest.
        """

        node_id = int(node_id)
        interact_time = float(interact_time)

        # searchsorted with side='left' gives the first index where time >= interact_time.
        # Therefore, [:i] contains only times < interact_time.
        i = np.searchsorted(
            self.nodes_neighbor_times[node_id],
            interact_time
        )

        return (
            self.nodes_neighbor_ids[node_id][:i],
            self.nodes_edge_ids[node_id][:i],
            self.nodes_neighbor_times[node_id][:i]
        )

    def get_neighbors_src(self, node_ids: np.ndarray,
                          node_interact_times: np.ndarray,
                          edge_ids: np.ndarray):
        """
        get historical neighbors of nodes in node_ids with interactions before the corresponding time in node_interact_times
        :param node_ids: ndarray, shape (batch_size, ) or (*, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ) or (*, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        nodes, edges, times, valid = self._recent_candidates(
            node_ids, edge_ids, self.hyper.N
        )
        return (
            nodes.masked_fill(~valid, 0),
            edges.masked_fill(~valid, 0),
            times.masked_fill(~valid, 0.0)
        )

    def get_neighbors_dst(
        self,
        node_ids: np.ndarray,
        target_skill: np.ndarray,
        node_interact_times: np.ndarray,
        edge_ids: np.ndarray
    ):
        """
        Get contextual historical neighbors.

        For each node in node_ids:
        1. Retrieve all historical neighbors before the current timestamp.
        2. Keep the last gamma * N neighbors.
        3. Re-rank them using time + concept relevance.
        4. Select the top N.
        5. Return them in chronological order.

        Important:
        target_skill[idx] should be the skill of the current target question.
        """
        
        N = self.hyper.N
        num_candidates = int(self.hyper.gamma * N)

        nodes, edges, times, valid = self._recent_candidates(
            node_ids, edge_ids, num_candidates
        )
        target_skill = torch.as_tensor(
            target_skill, device=self.device
        )
        historical_skills = self.node_skills_t[nodes]
        matching_skill = historical_skills == target_skill[:, None]

        # Skill 1 is the dataset sentinel indicating that the current record
        # has no concept in common with the history. Disable the contextual
        # contribution for the whole query (effective tau_c = 0), even when a
        # historical record also carries the sentinel value 1.
        contextual_enabled = target_skill != 1
        matching_skill = matching_skill & contextual_enabled[:, None]

        # Column 0 is oldest and the last column newest. Invalid left padding
        # receives -inf and can never displace a real historical interaction.
        ranks = torch.arange(
            num_candidates, 0, -1,
            dtype=torch.float32, device=self.device
        )
        decay = self.hyper.tau_t * (
            1.0 + self.hyper.tau_c * matching_skill.float()
        )
        scores = -(ranks[None, :] - 1.0) / decay
        scores = scores.masked_fill(~valid, float('-inf'))

        selected = torch.topk(scores, k=N, dim=1, sorted=False).indices
        selected_nodes = nodes.gather(1, selected)
        selected_edges = edges.gather(1, selected)
        selected_times = times.gather(1, selected)
        selected_valid = valid.gather(1, selected)

        # Preserve chronological output and left-padding. Invalid entries sort
        # first; valid entries are ordered by their historical timestamp.
        sort_key = selected_times.masked_fill(~selected_valid, float('-inf'))
        order = sort_key.argsort(dim=1)
        selected_nodes = selected_nodes.gather(1, order)
        selected_edges = selected_edges.gather(1, order)
        selected_times = selected_times.gather(1, order)
        selected_valid = selected_valid.gather(1, order)

        return (
            selected_nodes.masked_fill(~selected_valid, 0),
            selected_edges.masked_fill(~selected_valid, 0),
            selected_times.masked_fill(~selected_valid, 0.0)
        )
