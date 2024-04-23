use std::fs::File;
use std::io::{BufRead, BufReader};
use petgraph::graph::{Graph, NodeIndex};

pub struct graph_maker; 

impl graph_maker {
    fn parse_adjacency_matrix_from_file_path(&self, file_path: &str) -> Vec<Vec<f32>> {
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

    pub fn build_graph_from_adjacency_matrix(&self, adjacency_matrix: Vec<Vec<f32>>)  -> Graph<i32, f32> {
        let mut graph: Graph<i32, f32> = Graph::new();
        let mut node_indices: Vec<NodeIndex> = Vec::new();
        for i in 0.. adjacency_matrix.len() {
            let node_index = graph.add_node(0); 
            node_indices.push(node_index);
        }
        for i in 0.. adjacency_matrix.len() {
            for j in 0 .. adjacency_matrix[i].len() {
                if adjacency_matrix[i][j] != 0.0 {
                    let weight = adjacency_matrix[i][j];
                    graph.add_edge(node_indices[i], node_indices[j], weight);
                }
            }
        }
        return graph;
    }

    pub fn build_graph_from_file_path(&self, file_path: &str) -> Graph<i32, f32> {
            let adjacency_matrix = self.parse_adjacency_matrix_from_file_path(file_path);
            let graph = self.build_graph_from_adjacency_matrix(adjacency_matrix);
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
    let actual_matrix_1 = graph_maker.parse_adjacency_matrix_from_file_path("data/matrix_1.txt");
    let actual_matrix_2 = graph_maker.parse_adjacency_matrix_from_file_path("data/matrix_2.txt");
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
    let actual_graph = graph_maker.build_graph_from_adjacency_matrix(input_matrix);
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
    let actual_graph = graph_maker.build_graph_from_file_path("data/matrix_3.txt");
    let node_matcher = |_: &_, _: &_| true;
    let edge_matcher = |edge1: &f32, edge2: &f32| *edge1 == *edge2;
    assert!(is_isomorphic_matching(&expected_graph, &actual_graph, node_matcher, edge_matcher));
    }
}