use std::fs::File;
use std::io::{BufRead, BufReader};
use petgraph::graph::{Graph, NodeIndex};
use crate::run_manager::community_detection::Node;
use rayon::ThreadPoolBuilder;
pub struct graph_maker; 
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::sync::Mutex;
use std::sync::Arc;

impl graph_maker {
    fn parse_adjacency_matrix_from_file_path(&self, file_path: &str, thread: i32) -> Vec<Vec<f32>> {
        let mut adjacency_matrix:  Vec<Vec<f32>> = Vec::new();
        if let Ok(file) = File::open(file_path) {
            let reader = BufReader::new(file);
            for line in reader.lines() {             
                    let numbers: Vec<f32> = line
                        .unwrap()
                        .split_whitespace()
                        .map(|s| s.parse::<f32>())
                        .filter_map(Result::ok)
                        .collect();
                    adjacency_matrix.push(numbers);  
            }
        }
        return adjacency_matrix;
    }

    pub fn build_graph_from_adjacency_matrix(&self, adjacency_matrix: Vec<Vec<f32>>, thread: i32) -> Graph<Node, f32> {
        
    
        let graph: Arc<Mutex<Graph<Node, f32>>> = Arc::new(Mutex::new(Graph::new())); 
        let node_indices: Arc<Mutex<Vec<NodeIndex>>> = Arc::new(Mutex::new(Vec::new())); 
    
        (0..adjacency_matrix.len()).into_par_iter().for_each(|i| {
            let mut graph = graph.lock().unwrap(); 
            let mut node_indices = node_indices.lock().unwrap(); 
            let node_index = graph.add_node(Node::new(i.try_into().unwrap(), vec![i.try_into().unwrap()]));
            node_indices.push(node_index); 
        });
    
        let mut graph = Arc::try_unwrap(graph)
            .expect("Failed to unwrap Arc")
            .into_inner()
            .expect("Failed to get inner value of Mutex");
    
        let node_indices = Arc::try_unwrap(node_indices)
            .expect("Failed to unwrap Arc")
            .into_inner()
            .expect("Failed to get inner value of Mutex");
    
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
    

    pub fn build_graph_from_file_path(&self, file_path: &str, thread: i32) -> Graph<Node, f32> {
            let adjacency_matrix = self.parse_adjacency_matrix_from_file_path(file_path, thread);
            let graph = self.build_graph_from_adjacency_matrix(adjacency_matrix, thread);
            return graph;
    }

}

     


#[cfg(test)]
mod tests {
    use petgraph::graph::Graph;
    use petgraph::graph::NodeIndex;
    use petgraph::algo::is_isomorphic_matching;
    use crate::run_manager::graph_maker::graph_maker;
    
    #[test]
    fn test_adjacency_matrix_from_file() {
        let expected_matrix_1: Vec<Vec<f32>> = vec![
        vec![0.1, 0.7, 0.8],
        vec![0.8, 1.5, 0.0],
        vec![0.0, 0.8, 0.0],
    ];
    let expected_matrix_2: Vec<Vec<f32>> = vec![
        vec![0.1, 0.5, 0.9, 0.7, 1.4],
        vec![0.8, 1.5, 0.0, 0.7, 1.5],
        vec![0.0, 0.8, 0.0, 0.0, 0.1],
        vec![0.1, 0.4, 0.3, 0.2, 10.5],
        vec![11.4, 0.1, 0.3, 11.1, 0.1],
    ];
    let actual_matrix_1 = graph_maker.parse_adjacency_matrix_from_file_path("data/matrix_1.txt", 1);
    let actual_matrix_2 = graph_maker.parse_adjacency_matrix_from_file_path("data/matrix_2.txt", 1);
    assert_eq!(actual_matrix_1, expected_matrix_1);
    assert_eq!(actual_matrix_2, expected_matrix_2);
    }

    #[test]
    fn build_graph_from_adjacency_matrix_test() {
        let input_matrix: Vec<Vec<f32>> = vec![
        vec![0.1, 0.7, 0.8],
        vec![0.8, 1.5, 0.0],
        vec![0.0, 0.8, 0.0],
    ];
    let mut expected_graph: Graph<i32, f32> = Graph::new();
    let node_indices: Vec<NodeIndex> = input_matrix.iter().map(|_| expected_graph.add_node(0)).collect();
    expected_graph.add_edge(node_indices[0], node_indices[0], 0.1);
    expected_graph.add_edge(node_indices[0], node_indices[1], 0.7);
    expected_graph.add_edge(node_indices[0], node_indices[2], 0.8);
    expected_graph.add_edge(node_indices[1], node_indices[0], 0.8);
    expected_graph.add_edge(node_indices[1], node_indices[1], 1.5);
    expected_graph.add_edge(node_indices[2], node_indices[1], 0.8);
    let actual_graph = graph_maker.build_graph_from_adjacency_matrix(input_matrix, 1);
    let node_matcher = |_: &_, _: &_| true;
    let edge_matcher = |edge1: &f32, edge2: &f32| *edge1 == *edge2;
    assert!(is_isomorphic_matching(&expected_graph, &actual_graph, node_matcher, edge_matcher));
    }

    #[test]
    fn build_graph_from_file_path_test() {
        let input_matrix: Vec<Vec<f32>> = vec![
        vec![0.2, 0.8, 0.9],
        vec![0.9, 1.6, 0.0],
        vec![0.0, 0.9, 0.0],
    ];
    let mut expected_graph: Graph<i32, f32> = Graph::new();
    let node_indices: Vec<NodeIndex> = input_matrix.iter().map(|_| expected_graph.add_node(0)).collect();
    expected_graph.add_edge(node_indices[0], node_indices[0], 0.2);
    expected_graph.add_edge(node_indices[0], node_indices[1], 0.8);
    expected_graph.add_edge(node_indices[0], node_indices[2], 0.9);
    expected_graph.add_edge(node_indices[1], node_indices[0], 0.9);
    expected_graph.add_edge(node_indices[1], node_indices[1], 1.6);
    expected_graph.add_edge(node_indices[2], node_indices[1], 0.9);
    let actual_graph = graph_maker.build_graph_from_file_path("data/matrix_3.txt", 1);
    let node_matcher = |_: &_, _: &_| true;
    let edge_matcher = |edge1: &f32, edge2: &f32| *edge1 == *edge2;
    assert!(is_isomorphic_matching(&expected_graph, &actual_graph, node_matcher, edge_matcher));
    }
}