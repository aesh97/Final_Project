use std::collections::HashMap;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::Direction;
use petgraph::visit::EdgeRef;
use crate::run_manager::graph_maker::graph_maker;
use std::fmt;
use std::ops::Index;
use std::collections::HashSet;
use petgraph::visit::NodeRef;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use std::sync::Mutex;

#[derive(Debug, PartialEq, Clone)]
pub struct partition {
    node_indexes: Vec<NodeIndex>,
    current_partition: HashMap<NodeIndex, i32>,

}

impl Node {
    // Initializer function to create a new Node instance
    pub fn new(id: i32, indexes: Vec<i32>) -> Self {
        Node { id, indexes }
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    id: i32,
    indexes: Vec<i32>,
}

impl partition {
    pub fn new(node_indexes: Vec<NodeIndex>) -> Self {
        partition {
            node_indexes,
            current_partition: HashMap::new(),
        }   
    }

   


    pub fn make_node_singleton(&mut self, node: NodeIndex) {
        let maximum_community_number = self.current_partition
                                                .values()
                                                .max()
                                                .unwrap();
        self.current_partition.insert(node, maximum_community_number+1);
    }

    pub fn reasign_node_to_optimal_community(&mut self, node: NodeIndex, graph: &Graph<Node, f32>) -> f32 {
        let mut change_in_modularity = 0.0;
        let mut optimal_community =  self.get_node_partition(node).unwrap();
        for neighbor in graph.neighbors_directed(node, petgraph::Direction::Outgoing).collect::<Vec<_>>() {
            let community_of_neighbor = self.get_node_partition(neighbor).unwrap();
            let possible_change_in_modularity = self.compute_change_in_modularty(node, community_of_neighbor, &graph, 1);
            if possible_change_in_modularity > change_in_modularity {
                optimal_community = community_of_neighbor;
                change_in_modularity = possible_change_in_modularity;
            }
        }
        self.assign_node_to_cluster(node, optimal_community);
        return change_in_modularity;
    }

    fn remove_duplicates(vec: Vec<i32>, thread: i32) -> Vec<i32> {
        
        let set: HashSet<i32> = vec.into_par_iter().collect();
        set.into_par_iter().collect()
    }

    fn adjust_cluster_numbers(&mut self, threads: i32) {
        let mut cluster_numbers_sorted: Vec<i32> = Self::remove_duplicates(self.current_partition.values().cloned().collect(), threads);
        cluster_numbers_sorted.sort();
        
        for node in self.node_indexes.clone() {
            self.assign_node_to_cluster(node, cluster_numbers_sorted.iter().position(|&x| x == *self.current_partition.get(&node).unwrap()).unwrap() as i32);
        }
    }

    pub fn phase_2(&mut self, G: Graph<Node, f32>, threads: i32) -> Graph<Node, f32> {
        self.adjust_cluster_numbers(threads);
        let number_of_clusters = self.get_number_of_clusters(threads);
        let mut adjacency_matrix: Vec<Vec<f32>> = vec![vec![0.0; number_of_clusters.try_into().unwrap()]; number_of_clusters.try_into().unwrap()];
        for &node in &self.node_indexes {
            for edge in G.edges_directed(node, Direction::Outgoing) {
                let source = edge.source();
                let target = edge.target();
                if let Some(source_cluster) = self.get_node_partition(source) {
                    if let Some(target_cluster) = self.get_node_partition(target) {
                        if source_cluster != target_cluster {
                            adjacency_matrix[source_cluster as usize][target_cluster as usize] += edge.weight();
                        }
                    }
                }
            }
        }
        
        let mut graph: Graph<Node, f32> = Graph::new();
        let mut node_indices: Vec<NodeIndex> = Vec::new();
        for i in 0..adjacency_matrix.len() {
            let mut nodes_represented: Vec<i32> = vec![];
            for node_index in self.get_nodes_in_cluster(i.try_into().unwrap(), threads) {
                if let Some(node) = G.node_weight(node_index) {
                    for rep in &node.indexes {
                        nodes_represented.push(*rep);
                    }
                } 
            }
            let node_index = graph.add_node(Node::new(i.try_into().unwrap(), nodes_represented)); 
            node_indices.push(node_index);
        }
        for i in 0..adjacency_matrix.len() {
            for j in 0..adjacency_matrix[i].len() {
                if adjacency_matrix[i][j] != 0.0 {
                    let weight = adjacency_matrix[i][j];
                    graph.add_edge(node_indices[i], node_indices[j], weight);
                }
            }
        }
        graph
    }


    
    
    pub fn initialize_singleton(&mut self) {
        let partition_mutex = Mutex::new(&mut self.current_partition);
        self.node_indexes
            .par_iter()
            .enumerate()
            .for_each(|(i, node_index)| {
                let mut partition = partition_mutex.lock().unwrap();
                partition.insert(*node_index, i as i32);
            });
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

    pub fn get_number_of_clusters(&self, threads: i32) -> i32 {
        return Self::remove_duplicates(self.current_partition.values().cloned().collect(), threads).len().try_into().unwrap();
    }

    pub fn get_nodes_in_cluster(&self, cluster: i32, theads: i32) -> Vec<NodeIndex> {
        return self.node_indexes
            .clone()
            .into_par_iter()
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

    fn compute_change_in_modularty(&self, node: NodeIndex, temp_community: i32, graph: &Graph<Node, f32>, theads: i32) -> f32 {
        let mut m: f32 = 0.0;
        for edge in graph.edge_indices() {
            m += *graph.edge_weight(edge).unwrap();
        }
        
        m /= 2.0;
        if m == 0.0 {
            return 0.0;
        }
        let mut potential_community_nodes = self.get_nodes_in_cluster(temp_community, theads);
        let term_1: f32 = community_detection.sum_of_weights_from_node_to_community(node, &potential_community_nodes, graph);
        let term_2: f32 = community_detection.outward_degree_of_node(node, &graph) * community_detection.sum_of_in_going_edges_to_nodes_in_a_community(&potential_community_nodes, &graph);
        return term_1  / m- (term_2 / (2.0*m*m));
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
    fn sum_of_weights_from_node_to_community(&self, node: NodeIndex, community: &Vec<NodeIndex>, graph: &Graph<Node, f32>) -> f32 {
        let mut weight = 0.0;
        let mut outgoing_edges = graph.edges_directed(node, Direction::Outgoing);
        for edge in outgoing_edges {
            if community.contains(&edge.target()) {
                weight += edge.weight();
            }
        }
        return weight;
    }

    fn inward_degree_of_node(&self, node: NodeIndex, graph: &Graph<Node, f32>) -> f32 {
        return graph
        .edges_directed(node, Direction::Incoming)
        .map(|edge| *graph.edge_weight(edge.id()).unwrap_or(&0.0))
        .sum();
    }

    fn outward_degree_of_node(&self, node: NodeIndex, graph: &Graph<Node, f32>) -> f32 {
        return graph
        .edges_directed(node, Direction::Outgoing)
        .map(|edge| *graph.edge_weight(edge.id()).unwrap_or(&0.0))
        .sum();
    }

    fn sum_of_in_going_edges_to_nodes_in_a_community(&self, community: &Vec<NodeIndex>, graph: &Graph<Node, f32>) -> f32 {
        return community
            .iter()
            .map(|node| self.inward_degree_of_node(*node, &graph))
            .sum();
    }

    fn sum_of_out_going_edges_to_nodes_in_a_community(&self, community: &Vec<NodeIndex>, graph: &Graph<Node, f32>) -> f32 {
        return community
            .iter()
            .map(|node| self.outward_degree_of_node(*node, &graph))
            .sum();
    }

    fn phase_1(&self, G: &Graph<Node, f32>, thread: i32) -> (partition, f32) {
        let mut node_indices: Vec<NodeIndex> = G.node_indices().collect();
        let mut partition = partition::new(node_indices.clone());
        partition.initialize_singleton();
        let mut total_increase = 1000.0;
        let mut net_increase = 0.0;
        while total_increase > 0.1 {
            total_increase = 0.0;
            for node_index in &node_indices {
                let original_community: i32 = partition.get_node_partition(*node_index).unwrap();
                partition.make_node_singleton(*node_index);
                total_increase -= partition.compute_change_in_modularty(*node_index, original_community, &G, thread);
                total_increase += partition.reasign_node_to_optimal_community(*node_index, &G);
            }
            net_increase += total_increase;
        }
        partition.adjust_cluster_numbers(thread);
        return (partition, net_increase);
    }

    fn compute_modularity_from_singleton(&self, G: &Graph<Node, f32>) -> f32 {
        let mut m: f32 = 0.0;
        for edge in G.edge_indices() {
            m += *G.edge_weight(edge).unwrap();
        }
        let mut node_indices: Vec<NodeIndex> = G.node_indices().collect();
        let  modularity: f32 = node_indices
                            .iter()
                            .map(|node| -1.0* (self.inward_degree_of_node(*node, G) / m)* (self.inward_degree_of_node(*node, G) / m))
                            .sum();
        return modularity;
    }

    pub fn serial_louvain_algorithm(&self, G: &Graph<Node, f32>, threads: i32) -> (partition, f32) {
        
        let mut modularity = self.compute_modularity_from_singleton(G);
        let mut partition = self.phase_1(G, threads).0; 
    
        let mut graph = G.clone();
       
        loop {
            let new_graph = partition.phase_2(graph, threads);
            let (new_partition, total_increase) = self.phase_1(&new_graph, threads);
            
            partition = new_partition;
            modularity += total_increase;
    
           
            if total_increase.abs() < 0.1 {
                break;
            }
            graph = new_graph;
        }

       
        (partition, modularity)
    }
}

#[cfg(test)]
mod tests {
    use petgraph::algo::is_isomorphic_matching;
    use crate::run_manager::community_detection::partition;
    use petgraph::graph::{Graph, NodeIndex};
    use crate::run_manager::community_detection::community_detection;
    use std::collections::HashMap;
    use crate::run_manager::community_detection::Node;
    use crate::run_manager::graph_maker::graph_maker;
    

    #[test]
    fn create_singleton_partition_of_graph_test() {
        let mut graph: Graph<Node, f32> = Graph::new();
        let node_1 = graph.add_node(Node::new(0, vec![0]));
        let node_2 = graph.add_node(Node::new(0, vec![0]));
        let node_3 = graph.add_node(Node::new(0, vec![0]));
        let node_4 = graph.add_node(Node::new(0, vec![0]));
        let edge_1 = graph.add_edge(node_1, node_2, 0.5);
        let node_indeces: Vec<_> = graph.node_indices().collect();
        let mut partition: partition = partition::new(node_indeces.clone());
        partition::initialize_singleton(&mut partition);
        let mut i = 0;
        for node in node_indeces {
            let node_community = partition::get_node_partition(&partition, node).unwrap();
            assert_eq!(node_community, i);
            i += 1;
        }
    }

    #[test]
    fn assign_node_to_cluster_then_reasign_from_singleton_test() {
        let mut graph: Graph<Node, f32> = Graph::new();
        let node_1 = graph.add_node(Node::new(0, vec![0]));
        let node_2 = graph.add_node(Node::new(0, vec![0]));
        let edge_1 = graph.add_edge(node_1, node_2, 0.5);
        let node_indeces: Vec<_> = graph.node_indices().collect();
        let mut partition: partition = partition::new(node_indeces.clone());
        partition::initialize_singleton(&mut partition);
        partition::assign_node_to_cluster(&mut partition, node_1, 1);
        let node_1_community = partition::get_node_partition(&mut partition, node_1).unwrap();
        let node_2_community = partition::get_node_partition(&mut partition, node_2).unwrap();
        assert_eq!(node_1_community, 1);
        assert_eq!(node_2_community, 1);
        partition::assign_node_to_cluster(&mut partition, node_1, 0);
        partition::assign_node_to_cluster(&mut partition, node_2, 0);
        let node_1_community_final = partition::get_node_partition(&mut partition, node_1).unwrap();
        let node_2_community_final = partition::get_node_partition(&mut partition, node_2).unwrap();
        assert_eq!(node_1_community_final, 0);
        assert_eq!(node_2_community_final, 0);
    }

    #[test]
    fn assign_node_to_cluster_then_reasign_from_default_test() {
        let mut graph: Graph<Node, f32> = Graph::new();
        let node_1 = graph.add_node(Node::new(0, vec![0]));
        let node_2 = graph.add_node(Node::new(0, vec![0]));
        let edge_1 = graph.add_edge(node_1, node_2, 0.5);
        let node_indeces: Vec<_> = graph.node_indices().collect();
        let mut partition: partition = partition::new(node_indeces.clone());
        partition::assign_node_to_cluster(&mut partition, node_1, 1);
        let node_1_community = partition::get_node_partition(&mut partition, node_1).unwrap();
        assert_eq!(node_1_community, 1);
        partition::assign_node_to_cluster(&mut partition, node_1, 0);
        partition::assign_node_to_cluster(&mut partition, node_2, 0);
        let node_1_community_final = partition::get_node_partition(&mut partition, node_1).unwrap();
        let node_2_community_final = partition::get_node_partition(&mut partition, node_2).unwrap();
        assert_eq!(node_1_community_final, 0);
        assert_eq!(node_2_community_final, 0);
    }

    #[test]
    fn get_node_partition_test() {
        let mut graph: Graph<Node, f32> = Graph::new();
        let node_1 = graph.add_node(Node::new(0, vec![0]));
        let node_2 = graph.add_node(Node::new(0, vec![0]));
        let edge_1 = graph.add_edge(node_1, node_2, 0.5);
        let node_indeces: Vec<_> = graph.node_indices().collect();
        let mut partition: partition = partition::new(node_indeces.clone());
        assert_eq!(partition::get_node_partition(&mut partition, node_1), None);
        partition::assign_node_to_cluster(&mut partition, node_1, 0);
        assert_eq!(partition::get_node_partition(&mut partition, node_1), Some(0));
    }

    #[test]
    fn get_number_of_clusters_test() {
        let mut graph: Graph<Node, f32> = Graph::new();
        let node_1 = graph.add_node(Node::new(0, vec![0]));
        let node_2 = graph.add_node(Node::new(0, vec![0]));
        let edge_1 = graph.add_edge(node_1, node_2, 0.5);
        let node_indeces: Vec<_> = graph.node_indices().collect();
        let mut partition: partition = partition::new(node_indeces.clone());
        assert_eq!(partition::get_number_of_clusters(&mut partition, 1), 0);
        partition::assign_node_to_cluster(&mut partition, node_1, 0);
        assert_eq!(partition::get_number_of_clusters(&mut partition, 1), 1);
    }

    #[test]
    fn get_nodes_in_cluster_test() {
        let mut graph: Graph<Node, f32> = Graph::new();
        let node_1 = graph.add_node(Node::new(0, vec![0]));
        let node_2 = graph.add_node(Node::new(0, vec![0]));
        let node_3 = graph.add_node(Node::new(0, vec![0]));
        let node_4 = graph.add_node(Node::new(0, vec![0]));
        let edge_1 = graph.add_edge(node_1, node_2, 0.5);
        let node_indeces: Vec<_> = graph.node_indices().collect();
        let mut partition: partition = partition::new(node_indeces.clone());
        partition.assign_node_to_cluster(node_1, 0);
        partition.assign_node_to_cluster(node_2, 0);
        partition.assign_node_to_cluster(node_3, 2);
        partition.assign_node_to_cluster(node_4, 1);
        let expected_partition_0 = vec![node_1, node_2];
        let expected_partition_2 = vec![node_3];
        let expected_partition_1 = vec![node_4];
        let expected_partition_3 = vec![];
        let actual_partition_0 = partition::get_nodes_in_cluster(&partition, 0, 1);
        let actual_partition_2 = partition::get_nodes_in_cluster(&partition, 2, 1);
        let actual_partition_1 = partition::get_nodes_in_cluster(&partition, 1, 1);
        let actual_partition_3 = partition::get_nodes_in_cluster(&partition, 3, 1);
        assert_eq!(actual_partition_0, expected_partition_0);
        assert_eq!(actual_partition_1, expected_partition_1);
        assert_eq!(actual_partition_2, expected_partition_2);
        assert_eq!(actual_partition_3, expected_partition_3);
    }

    #[test]
    fn sum_of_weights_from_node_to_community_test() {
        let mut graph: Graph<Node, f32> = Graph::new();
        let node_A = graph.add_node(Node::new(0, vec![0]));
        let node_B = graph.add_node(Node::new(0, vec![0]));
        let node_C = graph.add_node(Node::new(0, vec![0]));
        let node_D = graph.add_node(Node::new(0, vec![0]));
        let node_E = graph.add_node(Node::new(0, vec![0]));
        let edge_1 = graph.add_edge(node_C, node_D, 0.7);
        let edge_2 = graph.add_edge(node_A, node_B, 0.7);
        let edge_3 = graph.add_edge(node_C, node_E, 1.1);
        let edge_4 = graph.add_edge(node_E, node_C, 0.4);
        let edge_5 = graph.add_edge(node_D, node_E, 10.1);
        let community = vec![node_D, node_E];
        let expected_output_C = 1.8;
        let expected_output_D = 10.1;
        let expected_output_E = 0.0;
        let actual_output_C = community_detection.sum_of_weights_from_node_to_community(node_C, &community, &graph);
        let actual_output_D = community_detection.sum_of_weights_from_node_to_community(node_D, &community, &graph);
        let actual_output_E = community_detection.sum_of_weights_from_node_to_community(node_E, &community, &graph);
        assert_eq!(expected_output_C, actual_output_C);
        assert_eq!(expected_output_D, actual_output_D);
        assert_eq!(expected_output_E, actual_output_E);
    }

    #[test]
    fn outward_inward_degree_of_node_test() {
        let mut graph: Graph<Node, f32> = Graph::new();
        let node_A = graph.add_node(Node::new(0, vec![0]));
        let node_B = graph.add_node(Node::new(0, vec![0]));
        let node_C = graph.add_node(Node::new(0, vec![0]));
        let edge_1 = graph.add_edge(node_C, node_A, 0.7);
        let edge_2 = graph.add_edge(node_A, node_B, 0.7);
        let edge_3 = graph.add_edge(node_C, node_B, 1.1);
        let edge_4 = graph.add_edge(node_A, node_C, 0.4);
        let expected_indward_degree_A = 0.7;
        let expected_indward_degree_B = 1.8;
        let expected_indward_degree_C = 0.4;
        let actual_inward_degree_A = community_detection.inward_degree_of_node(node_A, &graph);
        let actual_inward_degree_B = community_detection.inward_degree_of_node(node_B, &graph);
        let actual_inward_degree_C = community_detection.inward_degree_of_node(node_C, &graph);
        assert_eq!(expected_indward_degree_A, actual_inward_degree_A);
        assert_eq!(expected_indward_degree_B, actual_inward_degree_B);
        assert_eq!(expected_indward_degree_C, actual_inward_degree_C);
        let expected_outward_degree_A = 1.1;
        let expected_outward_degree_B = 0.0;
        let expected_outward_degree_C = 1.8;
        let actual_outward_degree_A = community_detection.outward_degree_of_node(node_A, &graph);
        let actual_outward_degree_B = community_detection.outward_degree_of_node(node_B, &graph);
        let actual_outward_degree_C = community_detection.outward_degree_of_node(node_C, &graph);
        assert_eq!(expected_outward_degree_A, actual_outward_degree_A);
        assert_eq!(expected_outward_degree_B, actual_outward_degree_B);
        assert_eq!(expected_outward_degree_C, actual_outward_degree_C);
    }

    #[test]
    fn sum_of_in_out_going_edges_to_nodes_in_a_community_test() {
        let mut graph: Graph<Node, f32> = Graph::new();
        let node_A = graph.add_node(Node::new(0, vec![0]));
        let node_B = graph.add_node(Node::new(0, vec![0]));
        let node_C = graph.add_node(Node::new(0, vec![0]));
        let node_D = graph.add_node(Node::new(0, vec![0]));
        let node_E = graph.add_node(Node::new(0, vec![0]));
        let node_F = graph.add_node(Node::new(0, vec![0]));
        let edge_1 = graph.add_edge(node_A, node_B, 0.7);
        let edge_2 = graph.add_edge(node_A, node_C, 0.4);
        let edge_3 = graph.add_edge(node_B, node_E, 0.7);
        let edge_4 = graph.add_edge(node_D, node_F, 10.3);
        let edge_5 = graph.add_edge(node_E, node_D, 0.1);
        let community = vec![node_C, node_E, node_D];
        let expected_incoming = 1.2;
        let expected_outgoing = 10.400001;
        let actual_incoming = community_detection.sum_of_in_going_edges_to_nodes_in_a_community(&community, &graph);
        let actual_outgoing: f32 = community_detection.sum_of_out_going_edges_to_nodes_in_a_community(&community, &graph);
        assert_eq!(expected_incoming, actual_incoming);
        assert_eq!(expected_outgoing, actual_outgoing);
    } 

    #[test]
    fn change_in_modularity_test() {
        let mut graph = Graph::<Node, f32>::new();
        let node_1 = graph.add_node(Node::new(0, vec![0]));
        let node_2 = graph.add_node(Node::new(0, vec![0]));
        let node_3 = graph.add_node(Node::new(0, vec![0]));
        let node_4 = graph.add_node(Node::new(0, vec![0]));
        let node_5 = graph.add_node(Node::new(0, vec![0]));
        let nodes = vec![node_1, node_2, node_3, node_4, node_5];
        let mut edge_weights: HashMap<(usize, usize), f32> = HashMap::new();
        graph.add_edge(node_1, node_2, 1.5);
        graph.add_edge(node_2, node_1, 1.5);
        graph.add_edge(node_1, node_3, 1.3333333333333333);
        graph.add_edge(node_3, node_1, 1.3333333333333333);
        graph.add_edge(node_1, node_4, 1.25);
        graph.add_edge(node_4, node_1, 1.25);       
        graph.add_edge(node_2, node_3, 1.6666666666666665);
        graph.add_edge(node_3, node_2, 1.6666666666666665);
        graph.add_edge(node_2, node_4, 1.5);
        graph.add_edge(node_4, node_2, 1.5);  
        graph.add_edge(node_3, node_4, 1.75);
        graph.add_edge(node_4, node_3, 1.75);  
        let mut partition: partition = partition::new(nodes.clone());
        partition.initialize_singleton();
        let actual_modularity_change_from_moving_node_2_to_cluster_0 = partition.compute_change_in_modularty(node_2, 0, &graph, 1);
        let expected_modularity_change = 0.04903979;
        assert_eq!(expected_modularity_change, actual_modularity_change_from_moving_node_2_to_cluster_0)
    }

    #[test]
    fn reasign_node_to_optimal_community_test() {
        let mut graph = Graph::<Node, f32>::new();
        let node_1 = graph.add_node(Node::new(0, vec![0]));
        let node_2 = graph.add_node(Node::new(0, vec![0]));
        let node_3 = graph.add_node(Node::new(0, vec![0]));
        let node_4 = graph.add_node(Node::new(0, vec![0]));
        let node_5 = graph.add_node(Node::new(0, vec![0]));
        let nodes = vec![node_1, node_2, node_3, node_4, node_5];
        graph.add_edge(node_1, node_2, 1.5);
        graph.add_edge(node_2, node_1, 1.5);
        graph.add_edge(node_1, node_3, 1.3333333333333333);
        graph.add_edge(node_3, node_1, 1.3333333333333333);
        graph.add_edge(node_1, node_4, 1.25);
        graph.add_edge(node_4, node_1, 1.25);       
        graph.add_edge(node_2, node_3, 1.6666666666666665);
        graph.add_edge(node_3, node_2, 1.6666666666666665);
        graph.add_edge(node_2, node_4, 1.5);
        graph.add_edge(node_4, node_2, 1.5);  
        graph.add_edge(node_3, node_4, 1.75);
        graph.add_edge(node_4, node_3, 1.75);  
        let mut partition: partition = partition::new(nodes.clone());
        partition.initialize_singleton();
        partition.assign_node_to_cluster(node_1, 0);
        partition.assign_node_to_cluster(node_2, 0);
        partition.assign_node_to_cluster(node_3, 1);
        partition.assign_node_to_cluster(node_4, 2);
        partition.assign_node_to_cluster(node_5, 3);
        partition.reasign_node_to_optimal_community(node_3, &graph);
        assert_eq!(partition.get_node_partition(node_1), Some(0));
        assert_eq!(partition.get_node_partition(node_2), Some(0));
        assert_eq!(partition.get_node_partition(node_3), Some(0));
        assert_eq!(partition.get_node_partition(node_4), Some(2));
        assert_eq!(partition.get_node_partition(node_5), Some(3));
    }

    #[test]
    fn make_node_singleton_test() {
        let mut graph = Graph::<Node, f32>::new();
        let node_1 = graph.add_node(Node::new(0, vec![0]));
        let node_2 = graph.add_node(Node::new(0, vec![0]));
        let node_3 = graph.add_node(Node::new(0, vec![0]));
        let node_4 = graph.add_node(Node::new(0, vec![0]));
        let node_5 = graph.add_node(Node::new(0, vec![0]));
        let nodes = vec![node_1, node_2, node_3, node_4, node_5];
        graph.add_edge(node_1, node_2, 1.5);
        graph.add_edge(node_2, node_1, 1.5);
        graph.add_edge(node_1, node_3, 1.3333333333333333);
        graph.add_edge(node_3, node_1, 1.3333333333333333);
        graph.add_edge(node_1, node_4, 1.25);
        graph.add_edge(node_4, node_1, 1.25);       
        graph.add_edge(node_2, node_3, 1.6666666666666665);
        graph.add_edge(node_3, node_2, 1.6666666666666665);
        graph.add_edge(node_2, node_4, 1.5);
        graph.add_edge(node_4, node_2, 1.5);  
        graph.add_edge(node_3, node_4, 1.75);
        graph.add_edge(node_4, node_3, 1.75);  
        let mut partition: partition = partition::new(nodes.clone());
        partition.initialize_singleton();
        partition.assign_node_to_cluster(node_1, 0);
        partition.assign_node_to_cluster(node_2, 0);
        partition.assign_node_to_cluster(node_3, 1);
        partition.assign_node_to_cluster(node_4, 2);
        partition.assign_node_to_cluster(node_5, 3);
        partition.make_node_singleton(node_2);
        let expected_community_number_for_node_2 = Some(4);
        let actual_community_number_for_node_2 = partition.get_node_partition(node_2);
        assert_eq!(actual_community_number_for_node_2,expected_community_number_for_node_2);
    }

    #[test]
    fn phase_1_test() {
        let mut graph = Graph::<Node, f32>::new();
        let node_1 = graph.add_node(Node::new(0, vec![0]));
        let node_2 = graph.add_node(Node::new(0, vec![0]));
        let node_3 = graph.add_node(Node::new(0, vec![0]));
        graph.add_edge(node_1, node_2, 11.7);
        graph.add_edge(node_2, node_1, 11.7);
        graph.add_edge(node_2, node_3, 10.4);
        graph.add_edge(node_3, node_2, 10.4);
        let nodes = vec![node_1, node_2, node_3];
        let mut partition: partition = partition::new(nodes.clone());
        partition.initialize_singleton();
        partition.assign_node_to_cluster(node_1, 1);
        let (mut actual_partition, actual_modularity) = community_detection.phase_1(&graph,1);
        let expected_modularity_increase = 0.37543252;
        assert_eq!(actual_modularity, expected_modularity_increase);
        let actual_node_1_cluster = actual_partition.get_node_partition(node_1);
        let actual_node_2_cluster = actual_partition.get_node_partition(node_2);
        let actual_node_3_cluster = actual_partition.get_node_partition(node_3);
        let expected_node_1_cluster = Some(0);
        let expected_node_2_cluster = Some(0);
        let expected_node_3_cluster = Some(0);
        assert_eq!(expected_node_1_cluster,actual_node_1_cluster);
        assert_eq!(expected_node_2_cluster,actual_node_2_cluster);
        assert_eq!(expected_node_3_cluster,actual_node_3_cluster);
    }

    #[test]
    fn compute_modularity_from_singleton_test() {
        let adjacency_matrix: Vec<Vec<f32>> = vec![
            vec![0.0, 0.9, 0.4, 0.2, 0.5, 0.0, 0.0],
            vec![0.9, 0.0, 0.5, 0.1, 0.3, 0.0, 0.0],
            vec![0.4, 0.5, 0.0, 0.9, 0.3, 0.0, 0.0],
            vec![0.2, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0],
            vec![0.5, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let mut G = graph_maker.build_graph_from_adjacency_matrix(adjacency_matrix, 1);
        let expected_singleton_modularity = -0.21267104;
        let actual_singleton_modularity = community_detection.compute_modularity_from_singleton(&G);
        assert_eq!(expected_singleton_modularity, actual_singleton_modularity);
    }

    #[test]
    fn adjust_cluster_numbers_test_1() {
        let mut graph = Graph::<Node, f32>::new();
        let node_1 = graph.add_node(Node::new(0, vec![0]));
        let node_2 = graph.add_node(Node::new(0, vec![0]));
        let node_3 = graph.add_node(Node::new(0, vec![0]));
        let node_4 = graph.add_node(Node::new(0, vec![0]));
        let node_5 = graph.add_node(Node::new(0, vec![0]));
        let nodes = vec![node_1, node_2, node_3, node_4, node_5];
        let mut partition: partition = partition::new(nodes.clone());
        partition.initialize_singleton();
        partition.assign_node_to_cluster(node_1, 0);
        partition.assign_node_to_cluster(node_2, 1);
        partition.assign_node_to_cluster(node_3, 12);
        partition.assign_node_to_cluster(node_4, 6);
        partition.assign_node_to_cluster(node_5, 100);
        partition.adjust_cluster_numbers(1);
        let expected_community_for_node_1 = Some(0);
        let expected_community_for_node_2 = Some(1);
        let expected_community_for_node_3 = Some(3);
        let expected_community_for_node_4 = Some(2);
        let expected_community_for_node_5 = Some(4);
        let actual_community_for_node_1 = partition.get_node_partition(node_1);
        let actual_community_for_node_2 = partition.get_node_partition(node_2);
        let actual_community_for_node_3 = partition.get_node_partition(node_3);
        let actual_community_for_node_4 = partition.get_node_partition(node_4);
        let actual_community_for_node_5 = partition.get_node_partition(node_5);
        assert_eq!(actual_community_for_node_1,expected_community_for_node_1);
        assert_eq!(actual_community_for_node_2,expected_community_for_node_2);
        assert_eq!(actual_community_for_node_3,expected_community_for_node_3);
        assert_eq!(actual_community_for_node_4,expected_community_for_node_4);
        assert_eq!(actual_community_for_node_5,expected_community_for_node_5);
    }

    #[test]
    fn adjust_cluster_numbers_test_2() {
        let mut graph = Graph::<Node, f32>::new();
        let node_1 = graph.add_node(Node::new(0, vec![0]));
        let node_2 = graph.add_node(Node::new(0, vec![0]));
        let node_3 = graph.add_node(Node::new(0, vec![0]));
        let node_4 = graph.add_node(Node::new(0, vec![0]));
        let node_5 = graph.add_node(Node::new(0, vec![0]));
        let nodes = vec![node_1, node_2, node_3, node_4, node_5];
        let mut partition: partition = partition::new(nodes.clone());
        partition.initialize_singleton();
        partition.assign_node_to_cluster(node_1, 0);
        partition.assign_node_to_cluster(node_2, 1);
        partition.assign_node_to_cluster(node_3, 12);
        partition.assign_node_to_cluster(node_4, 12);
        partition.assign_node_to_cluster(node_5, 6);
        partition.adjust_cluster_numbers(1);
        let expected_community_for_node_1 = Some(0);
        let expected_community_for_node_2 = Some(1);
        let expected_community_for_node_3 = Some(3);
        let expected_community_for_node_4 = Some(3);
        let expected_community_for_node_5 = Some(2);
        let actual_community_for_node_1 = partition.get_node_partition(node_1);
        let actual_community_for_node_2 = partition.get_node_partition(node_2);
        let actual_community_for_node_3 = partition.get_node_partition(node_3);
        let actual_community_for_node_4 = partition.get_node_partition(node_4);
        let actual_community_for_node_5 = partition.get_node_partition(node_5);
        assert_eq!(actual_community_for_node_1,expected_community_for_node_1);
        assert_eq!(actual_community_for_node_2,expected_community_for_node_2);
        assert_eq!(actual_community_for_node_3,expected_community_for_node_3);
        assert_eq!(actual_community_for_node_4,expected_community_for_node_4);
        assert_eq!(actual_community_for_node_5,expected_community_for_node_5);
    }

    #[test]
    fn phase_2_test() {
        let mut graph = Graph::<Node, f32>::new();
        let node_1 = graph.add_node(Node::new(0, vec![0]));
        let node_2 = graph.add_node(Node::new(0, vec![0]));
        let node_3 = graph.add_node(Node::new(0, vec![0]));
        let node_4 = graph.add_node(Node::new(0, vec![0]));
        let node_5 = graph.add_node(Node::new(0, vec![0]));
        let nodes = vec![node_1, node_2, node_3, node_4, node_5];
        let mut partition: partition = partition::new(nodes.clone());
        partition.initialize_singleton();
        graph.add_edge(node_1, node_2, 0.7);
        graph.add_edge(node_2, node_1, 0.7);
        graph.add_edge(node_3, node_2, 1.5);
        graph.add_edge(node_2, node_3, 1.5);
        graph.add_edge(node_4, node_2, 1.3);
        graph.add_edge(node_2, node_4, 1.3);
        graph.add_edge(node_4, node_5, 0.4);
        graph.add_edge(node_5, node_4, 0.4);
        partition.assign_node_to_cluster(node_1, 0);
        partition.assign_node_to_cluster(node_2, 1);
        partition.assign_node_to_cluster(node_3, 0);
        partition.assign_node_to_cluster(node_4, 2);
        partition.assign_node_to_cluster(node_5, 2);
        let mut expected_graph = Graph::<Node, f32>::new();
        let extected_node_1 =  expected_graph.add_node(Node::new(0, vec![0]));
        let extected_node_2 = expected_graph.add_node(Node::new(0, vec![0]));
        let extected_node_3 =  expected_graph.add_node(Node::new(0, vec![0]));
        expected_graph.add_edge(extected_node_1, extected_node_2, 2.2);
        expected_graph.add_edge(extected_node_2, extected_node_1, 2.2);
        expected_graph.add_edge(extected_node_3, extected_node_2, 1.3);
        expected_graph.add_edge(extected_node_2, extected_node_3, 1.3);
        let mut actual_graph = partition.phase_2(graph, 1);
        let node_matcher = |_: &_, _: &_| true;
        let edge_matcher = |edge1: &f32, edge2: &f32| *edge1 == *edge2;
        assert!(is_isomorphic_matching(&expected_graph, &actual_graph, node_matcher, edge_matcher));


    }
}