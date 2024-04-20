use std::fs::File;
use std::io::{BufRead, BufReader};

struct graph_maker(); 
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
}

#[cfg(test)]
mod tests {
    use crate::graph_maker::graph_maker;

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
    let actual_matrix_1 = graph_maker().parse_adjacency_matrix_from_file_path("data/matrix_1.txt");
    let actual_matrix_2 = graph_maker().parse_adjacency_matrix_from_file_path("data/matrix_2.txt");
    assert_eq!(actual_matrix_1, expected_matrix_1);
    assert_eq!(actual_matrix_2, expected_matrix_2);
    }
}