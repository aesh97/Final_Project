use std::collections::HashMap;
use petgraph::graph::{DiGraph, NodeIndex};
use graph_maker;

struct partition {
    node_indexes: Vec<NodeIndex>,
    current_partition: HashMap<NodeIndex, i32>,
}

impl partition {
    fn new(node_indexes: Vec<NodeIndex>) -> Self {
        partition {
            node_indexes: node_indexes,
            current_partition: HashMap::new(),
        }
    }
    
    pub fn initialize_singleton(&self) {
        let mut i = 0;
        for node_index in node_indexes {
            current_partition.insert(node_index, i);
            i += 1;
        }
    }

    pub fn assign_node_to_cluster(&self, node: NodeIndex, cluster: i32) {
        current_partition.insert(node, cluster);
    }

    pub fn get_node_partition(&self, node: NodeIndex) -> i32 {
        if let Some(value) = current_partition.get(node) {
            return value;
        } 
    }

    pub fn get_number_of_clusters(&self) -> i32 {
        return current_partition.values().cloned().max().unwrap_or_default();.
    }

    pub fn get_nodes_in_cluster(&self, cluster: i32) -> Vec<NodeIndex> {
        return node_indexes.filter_map(
            |node_index| {
                if current_partition.get(node_index) == cluster {
                    node_index
                } else{
                    None
                }
            },
        );
    }
}

struct community_detection;

impl community_detection {
    fn degree_of_node_in_community(&self, G: DiGraph<i32, f32>, node: NodeIndex, community: Vec<NodeIndex>) {
        let mut extended_community = community.to_vec();
        extended_community.push(node);
        let community_subgraph = G.filter_map(
            |node_idx| {
                if extended_community.contains(&node_idx) {
                    Some(node_idx)
                } else {
                    None
                }
            },
            |_, &edge| Some(edge),
        );
        let outgoing_degree: f32 = community_subgraph
        .edges_directed(node, Direction::Outgoing)
        .map(|edge_ref| *edge_ref.weight())
        .sum();
        let incoming_degree: f32 = community_subgraph
            .edges_directed(node, Direction::Incoming)
            .map(|edge_ref| *edge_ref.weight())
            .sum();
        return outgoing_degree + incoming_degree;
    }

    fn phase_1(&self, G: DiGraph<i32, f32>) -> (partition, f32) {
        let mut node_indices: Vec<NodeIndex> = G.node_indices().collect();
        let partition = partition::new(node_indices);
        partition::initialize_singleton(partition);
        let mut m = 0;
        for edge in G.raw_edges() {
            m += edge.weight;
        }
        let mut total_increase = 1000.0;
        while total_increase > 0.1 {
            total_increase = 0.0;
            for node_index in node_indices {
                if let Some(node_index) = G.neighbors_directed(node_index).collect::<Vec<NodeIndex>>() {
                    let community_of_node = partition.get_node_partition(node_index);
                    let mut maximum_modularity_increase = 0.0;
                    let mut best_community_option = community_of_node;
                    for neighbor in node_index {
                        let community_number_of_neighbor =  partition.get_node_partition(neighbor);
                        let community_of_neighbor = partition.get_nodes_in_cluster(community_number_of_neighbor);
                        let term_1 = self.degree_of_node_in_community(G, node_index, community_of_neighbor) / m;
                        let d_i_out = G
                            .edges_directed(node_index, Direction::Outgoing)
                            .map(|edge_ref| *edge_ref.weight())
                            .sum();
                        let d_i_in = G
                            .edges_directed(node_index, Direction::Incoming)
                            .map(|edge_ref| *edge_ref.weight())
                            .sum(); 
                        let mut sum_tot_in = 0.0;
                        let mut sum_tot_out = 0.0;
                        for edge in G.raw_edges() {
                            if community_of_neighbor.contains(G.target(edge).unwrap()) {
                                sum_tot_in += edge.weight;
                            }
                            if community_of_neighbor.contains(G.source(edge).unwrap()) {
                                sum_tot_out += edge.weight;
                            }
                        }    
                        let term_2 = (d_i_out*sum_tot_in + d_i_in*sum_tot_out) / (m * m);
                        let change_in_modularity = term_1 - term_2;
                        if change_in_modularity > maximum_modularity_increase {
                            maximum_modularity_increase = change_in_modularity;
                            best_community_option = community_number_of_neighbor;
                        }
                    }
                    total_increase += maximum_modularity_increase;
                    partition.assign_node_to_cluster(node_index, best_community_option);
                }
            }
        }
        return (partition, total_increase);
    }

    pub fn serial_louvain_algorithm(&self, G: DiGraph<i32, f32>) -> (partition, f32) {
        let mut m = 0;
        let mut node_indices: Vec<NodeIndex> = G.node_indices().collect();
        let mut A : Vec<i32> = [0; node_indexes.len()];
        for edge in G.raw_edges() {
            if G.target(edge) == G.source(edge) {
                A[G.target(edge)] = edge.weight;
            }
            m += edge.weight;
        }
        let mut modularity = 0.0;
        for i in 0 .. A.len() {
            let node = node_indeces[i];
            let d_i_out = G
                            .edges_directed(node, Direction::Outgoing)
                            .map(|edge_ref| *edge_ref.weight())
                            .sum();
            let d_i_in = G
                            .edges_directed(node, Direction::Incoming)
                            .map(|edge_ref| *edge_ref.weight())
                            .sum();
            modularity += (A[i] -(d_i_in* d_i_out)/m)
        }
        modularity /= m;
        let mut partition, mut total_increase = self.phase_1(G);
        modularity += total_increase;
        while total_increase > 0.01 {
            let number_of_clusters = partition.get_number_of_clusters();
            let mut graph: DiGraph<i32, f32> = DiGraph::new();
            let mut adjacency_matrix: Vec<Vec<f32>> = vec![vec![0.0; number_of_clusters]; number_of_clusters];
            for edge in G.raw_edges() {
                let target = G.target(edge).unwrap();
                let source = G.source(edge).unwrap();
                let target_cluster = partition.get_node_partition(target);
                let source_cluster = partition.get_node_partition(source);
                adjacency_matrix[source_cluster][target_cluster] += edge.weight;
            }

            let collapsed_graph = graph_maker().build_graph_from_adjacency_matrix(adjacency_matrix);
            partition, total_increase = self.phase_1(collapsed_graph);
            modularity += total_increase;
        }
        return (partition, modularity);
    }
}