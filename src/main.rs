//Made by Han_feng
mod lib;

use std::env;
use lib::vulkan_application::Vulkan_application;

fn main() {
    //Unify the work directory for the difference between cargo run and directly run the executable
    env::set_current_dir(env::current_exe().unwrap().parent().unwrap()).unwrap();

    let title = "Rust Vulkan Video Player".to_string();
    let size = (1920.0, 1080.0);
    
    let mut application = Vulkan_application::new(title, size);
    application.run();
}
