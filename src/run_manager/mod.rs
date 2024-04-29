pub mod community_detection;
pub mod graph_maker;






pub struct run_manager();

impl run_manager {
    pub fn run_on_file(file_path: &str, algorithm: &str) {
        
        let graph = graph_maker::graph_maker.build_graph_from_file_path(file_path);
        if algorithm == "Serial_Louvain" {
            let (partition, modularity) = community_detection::community_detection.serial_louvain_algorithm(&graph);
            println!("The resulting partition is: ");
            println!("{}", partition);
            println!("The resulting modularity is: ");
            println!("{}", modularity);
        } else {
            println!("This is not supported by the system yet");
        }
    }
}