use std::env;
mod run_manager;






fn main() {
    let args: Vec<String> = env::args().collect();
    println!("{}", args.len());
    if args.len() == 5 {
        let file_path = &args[1];
        let mode = &args[2];
        let algorithm_to_apply_to_graph = &args[3];
        let number_of_threads = &args[4];
        println!("{}", mode);
        if mode == "run" {
            println!("mode is run");
            run_manager::run_manager::run_on_file(file_path, algorithm_to_apply_to_graph);
        }
    } else {
        println!("There must be exactly 4 command line arguments. This is incorrectly formatted.");
    }
}