use std::collections::HashMap;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use petgraph::visit::EdgeRef;
use crate::run_manager::graph_maker::graph_maker;
use std::fmt;

#[derive(Debug, PartialEq, Clone)]
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

    pub fn reasign_node_to_optimal_community(&mut self, node: NodeIndex, graph: &DiGraph<i32, f32>) -> f32 {
        let mut change_in_modularity = 0.0;
        let mut optimal_community =  self.get_node_partition(node).unwrap();
        for neighbor in graph.neighbors_directed(node, petgraph::Direction::Outgoing).collect::<Vec<_>>() {
            let community_of_neighbor = self.get_node_partition(neighbor).unwrap();
            let possible_change_in_modularity = self.compute_change_in_modularty(node, community_of_neighbor, &graph);
            if possible_change_in_modularity > change_in_modularity {
                optimal_community = community_of_neighbor;
                change_in_modularity = possible_change_in_modularity;
            }
        }
        self.assign_node_to_cluster(node, optimal_community);
        return change_in_modularity;
    }
    
    pub fn initialize_singleton(&mut self) {
        let mut i = 0;
        for node_index in &self.node_indexes {
            self.current_partition.insert(*node_index, i);
            i += 1;
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
        return self.current_partition.values().cloned().len().try_into().unwrap();
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

    fn compute_change_in_modularty(&self, node: NodeIndex, temp_community: i32, graph: &DiGraph<i32, f32>) -> f32 {
        let mut m: f32 = 0.0;
        for edge in graph.edge_indices() {
            m += *graph.edge_weight(edge).unwrap();
        }
        


        println!("{}", m);
        let mut potential_community_nodes = self.get_nodes_in_cluster(temp_community);
        potential_community_nodes.push(node);

        println!("{}", potential_community_nodes.len());
        let term_1: f32 = community_detection.sum_of_weights_from_node_to_community(node, &potential_community_nodes, graph) as f32 / m as f32;
        println!("{}", term_1);
        let term_2: f32 = community_detection.outward_degree_of_node(node, &graph) * community_detection.sum_of_in_going_edges_to_nodes_in_a_community(&potential_community_nodes, &graph);
        println!("{}", term_2);
        let term_3: f32 = community_detection.inward_degree_of_node(node, &graph) * community_detection.sum_of_out_going_edges_to_nodes_in_a_community(&potential_community_nodes, &graph);
        println!("{}", term_3);
        return term_1 - ((term_2+term_3) / (m*m));
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
    fn sum_of_weights_from_node_to_community(&self, node: NodeIndex, community: &Vec<NodeIndex>, graph: &DiGraph<i32, f32>) -> f32 {
        let mut weight = 0.0;
        let mut outgoing_edges = graph.edges_directed(node, Direction::Outgoing);
        for edge in outgoing_edges {
            if community.contains(&edge.target()) {
                weight += edge.weight();
            }
        }
        return weight;
    }

    fn inward_degree_of_node(&self, node: NodeIndex, graph: &DiGraph<i32, f32>) -> f32 {
        return graph
        .edges_directed(node, Direction::Incoming)
        .map(|edge| *graph.edge_weight(edge.id()).unwrap_or(&0.0))
        .sum();
    }

    fn outward_degree_of_node(&self, node: NodeIndex, graph: &DiGraph<i32, f32>) -> f32 {
        return graph
        .edges_directed(node, Direction::Outgoing)
        .map(|edge| *graph.edge_weight(edge.id()).unwrap_or(&0.0))
        .sum();
    }

    fn sum_of_in_going_edges_to_nodes_in_a_community(&self, community: &Vec<NodeIndex>, graph: &DiGraph<i32, f32>) -> f32 {
        return community
            .iter()
            .map(|node| self.inward_degree_of_node(*node, &graph))
            .sum();
    }

    fn sum_of_out_going_edges_to_nodes_in_a_community(&self, community: &Vec<NodeIndex>, graph: &DiGraph<i32, f32>) -> f32 {
        return community
            .iter()
            .map(|node| self.outward_degree_of_node(*node, &graph))
            .sum();
    }

    fn phase_1(&self, G: &DiGraph<i32, f32>) -> (partition, f32) {
        let mut node_indices: Vec<NodeIndex> = G.node_indices().collect();
        let mut partition = partition::new(node_indices.clone());
        partition.initialize_singleton();
        let mut total_increase = 1000.0;
        while total_increase > 0.1 {
            total_increase = 0.0;
            for node_index in &node_indices {
                total_increase += partition.reasign_node_to_optimal_community(*node_index, &G);
            }
        }
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
        let (mut partition, mut total_increase) = self.phase_1(&G);
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
            (partition, total_increase) = self.phase_1(&collapsed_graph);
            modularity += total_increase;
        }
        return (partition, modularity);
    }
}

#[cfg(test)]
mod tests {
    use crate::run_manager::community_detection::partition;
    use petgraph::graph::{DiGraph, NodeIndex};
    use crate::run_manager::community_detection::community_detection;
    use std::collections::HashMap;

    #[test]
    fn create_singleton_partition_of_graph_test() {
        let mut graph: DiGraph<i32, f32> = DiGraph::new();
        let node_1 = graph.add_node(0);
        let node_2 = graph.add_node(0);
        let node_3 = graph.add_node(0);
        let node_4 = graph.add_node(0);
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
        let mut graph: DiGraph<i32, f32> = DiGraph::new();
        let node_1 = graph.add_node(0);
        let node_2 = graph.add_node(0);
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
        let mut graph: DiGraph<i32, f32> = DiGraph::new();
        let node_1 = graph.add_node(0);
        let node_2 = graph.add_node(0);
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
        let mut graph: DiGraph<i32, f32> = DiGraph::new();
        let node_1 = graph.add_node(0);
        let node_2 = graph.add_node(0);
        let edge_1 = graph.add_edge(node_1, node_2, 0.5);
        let node_indeces: Vec<_> = graph.node_indices().collect();
        let mut partition: partition = partition::new(node_indeces.clone());
        assert_eq!(partition::get_node_partition(&mut partition, node_1), None);
        partition::assign_node_to_cluster(&mut partition, node_1, 0);
        assert_eq!(partition::get_node_partition(&mut partition, node_1), Some(0));
    }

    #[test]
    fn get_number_of_clusters_test() {
        let mut graph: DiGraph<i32, f32> = DiGraph::new();
        let node_1 = graph.add_node(0);
        let node_2 = graph.add_node(0);
        let edge_1 = graph.add_edge(node_1, node_2, 0.5);
        let node_indeces: Vec<_> = graph.node_indices().collect();
        let mut partition: partition = partition::new(node_indeces.clone());
        assert_eq!(partition::get_number_of_clusters(&mut partition), 0);
        partition::assign_node_to_cluster(&mut partition, node_1, 0);
        assert_eq!(partition::get_number_of_clusters(&mut partition), 1);
    }

    #[test]
    fn get_nodes_in_cluster_test() {
        let mut graph: DiGraph<i32, f32> = DiGraph::new();
        let node_1 = graph.add_node(0);
        let node_2 = graph.add_node(0);
        let node_3 = graph.add_node(0);
        let node_4 = graph.add_node(0);
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
        let actual_partition_0 = partition::get_nodes_in_cluster(&partition, 0);
        let actual_partition_2 = partition::get_nodes_in_cluster(&partition, 2);
        let actual_partition_1 = partition::get_nodes_in_cluster(&partition, 1);
        let actual_partition_3 = partition::get_nodes_in_cluster(&partition, 3);
        assert_eq!(actual_partition_0, expected_partition_0);
        assert_eq!(actual_partition_1, expected_partition_1);
        assert_eq!(actual_partition_2, expected_partition_2);
        assert_eq!(actual_partition_3, expected_partition_3);
    }

    #[test]
    fn sum_of_weights_from_node_to_community_test() {
        let mut graph: DiGraph<i32, f32> = DiGraph::new();
        let node_A = graph.add_node(0);
        let node_B = graph.add_node(0);
        let node_C = graph.add_node(0);
        let node_D = graph.add_node(0);
        let node_E = graph.add_node(0);
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
        let mut graph: DiGraph<i32, f32> = DiGraph::new();
        let node_A = graph.add_node(0);
        let node_B = graph.add_node(0);
        let node_C = graph.add_node(0);
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
        let mut graph: DiGraph<i32, f32> = DiGraph::new();
        let node_A = graph.add_node(0);
        let node_B = graph.add_node(0);
        let node_C = graph.add_node(0);
        let node_D = graph.add_node(0);
        let node_E = graph.add_node(0);
        let node_F = graph.add_node(0);
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
        let mut graph = DiGraph::<i32, f32>::new();
        let node_1 = graph.add_node(0);
        let node_2 = graph.add_node(0);
        let node_3 = graph.add_node(0);
        let node_4 = graph.add_node(0);
        let node_5 = graph.add_node(0);
        let nodes = vec![node_1, node_2, node_3, node_4, node_5];
        let mut edge_weights: HashMap<(usize, usize), f32> = HashMap::new();
        

        graph.add_edge(node_1, node_2, 1.5);
        graph.add_edge(node_1, node_3, 1.3333333333333333);
        graph.add_edge(node_1, node_4, 1.25);
        graph.add_edge(node_2, node_1, 3.0);
        graph.add_edge(node_2, node_3, 1.6666666666666665);
        graph.add_edge(node_2, node_4, 1.5);
        graph.add_edge(node_3, node_1, 4.0);
        graph.add_edge(node_3, node_2, 2.5);
        graph.add_edge(node_3, node_4, 1.75);
        graph.add_edge(node_4, node_1, 5.0);
        graph.add_edge(node_4, node_2, 3.0);
        graph.add_edge(node_4, node_3, 2.333333333333333);
        
        
        let mut partition: partition = partition::new(nodes.clone());
        partition.initialize_singleton();
        

        let actual_modularity_change_from_moving_node_4_to_cluster_1 = partition.compute_change_in_modularty(node_1, 1, &graph);
        let expected_modularity_change = 0.03267733636272513;
        assert_eq!(expected_modularity_change, actual_modularity_change_from_moving_node_4_to_cluster_1)

    }
}