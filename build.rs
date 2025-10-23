use std::{env, fs};
use std::path::Path;

fn main() {
    let root_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let profile = env::var("PROFILE").unwrap();
    let out_path = Path::new(&root_dir).join("target").join(profile.as_str()).join("shaders");
    fs::create_dir_all(&out_path).unwrap();
    
    for entry in fs::read_dir(Path::new(&root_dir).join("shaders/output")).unwrap() {
        let source_path = entry.unwrap().path();
        let destination_path = out_path.join(source_path.file_name().unwrap());
        
        fs::copy(&source_path, &destination_path).unwrap();
        eprintln!("Copied {:?} to {:?}", source_path, destination_path);
    }

    // Note: I made a script to help me pre-compile the shader files, and then store the bytecodes to the output directory
    // You can make your own ones! :)
    println!("cargo:rerun-if-changed=shaders/output");
}