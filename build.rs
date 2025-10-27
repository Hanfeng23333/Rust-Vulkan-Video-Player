use std::{env, fs};
use std::path::Path;

fn main() {
    println!("cargo:rustc-link-lib=strmiids");
    println!("cargo:rustc-link-lib=mfuuid");

    let root_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let profile = env::var("PROFILE").unwrap();

    let root_path = Path::new(&root_dir);

    //Copy assets
    copy_dir_all(root_path.join("assets"), root_path.join("target").join(profile.as_str()).join("assets"));


    // Note: I made a script to help me pre-compile the shader files, and then store the bytecodes to the output directory
    // You can make your own ones! :)
    // I suggest that you can make your own assets folder, and all the used path can be found in the code :)
    println!("cargo:rerun-if-changed=assets");
}

fn copy_dir_all(src: impl AsRef<Path>, dst: impl AsRef<Path>) {
    fs::create_dir_all(&dst).unwrap();
    for entry in fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let ty = entry.file_type().unwrap();
        if ty.is_dir() {
            copy_dir_all(entry.path(), dst.as_ref().join(entry.file_name()));
        } else {
            fs::copy(entry.path(), dst.as_ref().join(entry.file_name())).unwrap();
        }
    }
}