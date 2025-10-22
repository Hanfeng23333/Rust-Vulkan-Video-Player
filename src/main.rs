mod lib;

use lib::vulkan_application::Vulkan_application;

fn main() {
    let application = Vulkan_application::new("Hello Vulkano".to_string());
    println!("Hello, vulkan!");
}
