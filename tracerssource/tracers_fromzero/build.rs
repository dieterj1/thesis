use std::process::Command;

fn main() {
    let output = Command::new("ldconfig")
        .arg("-p")
        .output()
        .expect("Failed to execute ldconfig");

    let stdout = String::from_utf8(output.stdout).expect("Failed to read ldconfig output");

    if stdout.contains("libopenblas") {
        println!("cargo:rustc-cfg=has_libopenblas");
        println!("cargo:rustc-link-lib=openblas");
    }
}
