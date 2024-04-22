use std::collections::HashMap;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use petgraph::visit::EdgeRef;
use crate::run_manager::graph_maker::graph_maker;
use std::fmt;

pub struct partition {
    node_indexes: Vec<NodeIndex>,
    current_partition: HashMap<NodeIndex, i32>,
}

impl partition {
    pub fn new(node_indexes: Vec<NodeIndex>) -> Self {
        partition {
            node_indexes,
            current_partition: HashMap::new(),
        }
    }
    
    pub fn initialize_singleton(&mut self) {
        let mut i = 0;
        for node_index in &self.node_indexes {
            self.current_partition.insert(*node_index, i);
            i += 1;
            println!("{}", i);
        }
    }

    pub fn assign_node_to_cluster(&mut self, node: NodeIndex, cluster: i32) {
        self.current_partition.insert(node, cluster);
    }

    pub fn get_node_partition(&self, node: NodeIndex) -> Option<i32> {
        if let Some(value) = self.current_partition.get(&node) {
            return Some(*value);
        } else {
            return None;
        }
    }

    pub fn get_number_of_clusters(&self) -> i32 {
        return self.current_partition.values().cloned().max().unwrap_or_default();
    }

    pub fn get_nodes_in_cluster(&self, cluster: i32) -> Vec<NodeIndex> {
        return self.node_indexes
            .clone()
            .into_iter()
            .filter_map(
                |node_index| {
                    if self.current_partition.get(&node_index) == Some(&cluster) {
                        Some(node_index)
                    } else{
                        None
                    }
                },
            )
            .collect();
    }
}

impl fmt::Display for partition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Node Indexes: ")?;
        for (index, node_index) in self.node_indexes.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", index)?;
        }
        write!(f, "\nCurrent Partition:\n")?;
        for (node_index, cluster) in &self.current_partition {
            write!(f, "Node {:?} -> Cluster {}\n", node_index, cluster)?;
        }
        Ok(())
    }
}


pub struct community_detection;

impl community_detection {
    fn degree_of_node_in_community(&self, G: &DiGraph<i32, f32>, node: NodeIndex, community: &Vec<NodeIndex>) -> f32 {
        let mut extended_community = community.clone();
        extended_community.push(node);
        let community_subgraph = G.filter_map(
            |node_idx, _| {
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
        let mut partition = partition::new(node_indices.clone());
        partition.initialize_singleton();
        let mut m = 0.0;
        for edge in G.raw_edges() {
            m += edge.weight;
        }
        println!("Made it here");
        let mut total_increase = 1000.0;
        while total_increase > 0.1 {
            total_increase = 0.0;
            //NOTE: this loop fails to update modularity correctly, and also it fails to update clusters of nodes 
            //NEEDS EXTENSIVE UNIT TESTING FOR:
            //partition.get_node_partition, .get_nodes_in_cluster, degree_of_node_in_community
            for node_index in &node_indices {
                let community_of_node = partition.get_node_partition(*node_index);
                let mut maximum_modularity_increase = 0.0;
                let mut best_community_option = community_of_node;
                for neighbor in G.neighbors_directed(*node_index, petgraph::Direction::Outgoing).collect::<Vec<_>>() {
                    let community_number_of_neighbor =  partition.get_node_partition(neighbor);
                    let community_of_neighbor = partition.get_nodes_in_cluster(community_number_of_neighbor.unwrap()).clone();
                    let term_1 = self.degree_of_node_in_community(&G, *node_index, &community_of_neighbor) / m;
                    let d_i_out: f32 = G
                        .edges_directed(*node_index, Direction::Outgoing)
                        .map(|edge_ref| *edge_ref.weight())
                        .sum();
                    let d_i_in: f32 = G
                        .edges_directed(*node_index, Direction::Incoming)
                        .map(|edge_ref| *edge_ref.weight())
                        .sum(); 
                    let mut sum_tot_in = 0.0;
                    let mut sum_tot_out = 0.0;
                    for edge in G.edges(*node_index) {
                        if community_of_neighbor.contains(&edge.target()) {
                            sum_tot_in += *edge.weight();
                        }
                        let source_index = edge.source();
                        if community_of_neighbor.contains(&source_index) {
                            sum_tot_out += *edge.weight();
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
                partition.assign_node_to_cluster(*node_index, best_community_option.unwrap());
            }
            //println!("Made it to the end of the loop !!!");

            
        }
        println!("finshed phase 1");
        return (partition, total_increase);
    }

    pub fn serial_louvain_algorithm(&self, G: &DiGraph<i32, f32>) -> (partition, f32) {
        let mut m = 0.0;
        let mut node_indices: Vec<NodeIndex> = G.node_indices().collect();
        let num_nodes = node_indices.len();
        let mut A : Vec<f32> = vec![0.0; num_nodes];
        for edge in G.raw_edges() {
            let source_index = node_indices.iter().position(|&x| x == edge.source()).unwrap();
            let target_index = node_indices.iter().position(|&x| x == edge.target()).unwrap();
            if source_index == target_index {
                A[target_index] = edge.weight;
            }
            m += edge.weight;
        }
        let mut modularity = 0.0;
        for i in 0 .. A.len() {
            let node = node_indices[i];
            let d_i_out: f32 = G
                            .edges_directed(node, Direction::Outgoing)
                            .map(|edge_ref| *edge_ref.weight())
                            .sum();
            let d_i_in: f32 = G
                            .edges_directed(node, Direction::Incoming)
                            .map(|edge_ref| *edge_ref.weight())
                            .sum();
            modularity += (A[i] -(d_i_in* d_i_out)/m)
        }
        modularity /= m;
        let (mut partition, mut total_increase) = self.phase_1(G.clone());
        modularity += total_increase;
        while total_increase > 0.01 {
            let number_of_clusters = partition.get_number_of_clusters();
            let mut adjacency_matrix: Vec<Vec<f32>> = vec![vec![0.0; number_of_clusters as usize]; number_of_clusters as usize];
            for edge in G.raw_edges() {
                let target_cluster = partition.get_node_partition(edge.target()).unwrap();
                let source_cluster = partition.get_node_partition(edge.source()).unwrap();
                adjacency_matrix[source_cluster as usize][target_cluster as usize] += edge.weight;
            }
            let collapsed_graph = graph_maker.build_graph_from_adjacency_matrix(adjacency_matrix);
            (partition, total_increase) = self.phase_1(collapsed_graph);
            modularity += total_increase;
            println!("Current Modularity: {}", modularity);
        }
        return (partition, modularity);
    }
}
