use std::env;

mod graph_maker;
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() == 4 {
        let file_path = &args[1];
        let make_graph = &args[2];
        let algorithm_to_apply_to_graph = &args[3];
        let number_of_threads = &args[4];
    } else {
        println!("There must be exactly 4 command line arguments. This is incorrectly formatted.");
    }
}