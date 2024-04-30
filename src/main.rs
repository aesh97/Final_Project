use std::env;
mod run_manager;
use rayon::ThreadPoolBuilder;
use std::time::{Instant, Duration};






fn main() {
    let start_time = Instant::now();
    let args: Vec<String> = env::args().collect();
    if args.len() == 5 {
        let file_path = &args[1];
        let mode = &args[2];
        let algorithm_to_apply_to_graph = &args[3];
        let number_of_threads = &args[4];
        if mode == "run" {
            let pool = ThreadPoolBuilder::new()
            .num_threads(number_of_threads.parse::<i32>().unwrap().try_into().unwrap())
            .build()
            .expect("Failed to create thread pool");
            run_manager::run_manager::run_on_file(file_path, algorithm_to_apply_to_graph, number_of_threads.parse::<i32>().unwrap());
        }
    } else {
        println!("There must be exactly 4 command line arguments. This is incorrectly formatted.");
    }
    let end_time = Instant::now();
    println!("Elapsed time: {:?}", end_time - start_time);

}