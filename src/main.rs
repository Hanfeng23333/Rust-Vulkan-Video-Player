//Made by Han_feng
mod lib;

use lib::vulkan_application::Vulkan_application;

fn main() {
    let title = "Rust Vulkan Video Player".to_string();
    let size = (800.0, 600.0);
    
    let mut application = Vulkan_application::new(title, size);
    application.run();
}
