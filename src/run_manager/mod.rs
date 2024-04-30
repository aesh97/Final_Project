pub mod community_detection;
pub mod graph_maker;






pub struct run_manager();

impl run_manager {
    pub fn run_on_file(file_path: &str, algorithm: &str, thread: i32) {
        
        let graph = graph_maker::graph_maker.build_graph_from_file_path(file_path, thread);
        if algorithm == "Serial_Louvain" {
            let (partition, modularity) = community_detection::community_detection.serial_louvain_algorithm(&graph, thread);
            println!("Final Partition: {:?}", partition);
            println!("Final Modularity: {}", modularity);

        } else {
            println!("This is not supported by the system yet");
        }
    }
}