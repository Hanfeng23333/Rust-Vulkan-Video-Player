//Made by Han_feng
mod lib;

use lib::Media_handler::Media_handler;

fn main() {
    //Unify the work directory for the difference between cargo run and directly run the executable
    std::env::set_current_dir(std::env::current_exe().unwrap().parent().unwrap()).unwrap();

    let args: Vec<String> = std::env::args().collect();

    let mut video_path = "assets/videos/video.mp4".to_string();

    if args.len() == 2{
        video_path = args[1].clone();
    }
    else if args.len() > 2{
        panic!("Usage: executable_name.exe [video_path] (If you don't provide the path, the default value is \"assets/videos/video.mp4\")");
    }

    let mut media_handler = Media_handler::new();
    media_handler.run(video_path);
}
