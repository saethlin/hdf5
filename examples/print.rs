fn main() -> Result<(), Box<dyn std::error::Error>> {
    let filename = std::env::args().nth(1).expect("Need a file path to open");
    let file = hdf5::read(&filename)?;
    println!("{:?}: {:#?}", filename, file);
    Ok(())
}
