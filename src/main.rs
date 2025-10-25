//Made by Han_feng
mod lib;

use lib::Media_handler::Media_handler;

fn main() {
    //Unify the work directory for the difference between cargo run and directly run the executable
    std::env::set_current_dir(std::env::current_exe().unwrap().parent().unwrap()).unwrap();

    let mut media_handler = Media_handler::new();
    media_handler.run("assets/videos/01.mp4".to_string());
}
